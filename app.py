from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import dlib
import tensorflow as tf
from pygame import mixer

app = Flask(__name__)

mixer.init()
mixer.music.load('./utils/alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("utils/shape_predictor_68_face_landmarks.dat")

model_eyes = tf.keras.models.load_model('sleep_detection/sleep_model_6.h5')
model_mouth = tf.keras.models.load_model('yawn_detection/yawn_model_1.h5')


def detect_drowsiness(image):
    # image = cv.resize(image, ())
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    prediction_left_eye = 0
    prediction_right_eye = 0
    prediction_mouth = 0

    eye_score = 0
    mouth_score = 0

    detect = detector(gray_img, 1)

    if(len(detect) > 0):
        shape = predictor(gray_img, detect[0])

        # left eye points
        xle1 = shape.part(17).x
        xle2 = shape.part(21).x

        yle1 = shape.part(19).y
        yle2 = shape.part(29).y

        # right eye points
        xre1 = shape.part(22).x
        xre2 = shape.part(26).x

        yre1 = shape.part(24).y
        yre2 = shape.part(29).y

        # mouth points
        xm1 = shape.part(36).x
        xm2 = shape.part(45).x

        ym1 = shape.part(33).y
        ym2 = shape.part(6).y

        # extracting the region of interests 
        left_eye_roi = gray_img[yle1:yle2, xle1:xle2]
        right_eye_roi = gray_img[yre1:yre2, xre1:xre2]
        mouth_roi = gray_img[ym1:ym2, xm1:xm2]

        # final left eye to feed to model 
        if left_eye_roi.size:
            final_left_eye = cv.resize(left_eye_roi, (64,64))
            final_left_eye = np.expand_dims(final_left_eye, axis=0)
            final_left_eye = final_left_eye/255.0
            prediction_left_eye = model_eyes.predict(final_left_eye)

        # final right eye to feed to model 
        if right_eye_roi.size:
            final_right_eye = cv.resize(right_eye_roi, (64,64))
            final_right_eye = np.expand_dims(final_right_eye, axis=0)
            final_right_eye = final_right_eye/255.0
            prediction_right_eye = model_eyes.predict(final_right_eye)

        # final mouth to feed to model 
        if mouth_roi.size:
            final_mouth = cv.resize(mouth_roi, (64,64))
            final_mouth = np.expand_dims(final_mouth, axis=0)
            final_mouth = final_mouth/255.0
            prediction_mouth = model_mouth.predict(final_mouth)

        
        # scores for eye and mouth individually 
        eye_score = (prediction_left_eye+prediction_right_eye)/2
        eye_score = prediction_left_eye
        eye_score = 1-eye_score

        mouth_score = prediction_mouth

        # combined score 
        total_score = (eye_score+mouth_score)/2
        # total_score = mouth_score
    else:
        total_score = 0
    
    scores = []
    scores.append(eye_score)
    scores.append(mouth_score)
    scores.append(total_score)

    return scores





@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    frame_check = 20
    cap = cv.VideoCapture(0)
    flag = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fr = cv.resize(frame, (240, 180))
        scores = detect_drowsiness(fr)
        eyeScore = scores[0]
        mouthScore = scores[1]
        score = scores[2]
        if score > 0.4:
            cv.putText(frame, f'Total: {score}', (10,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv.putText(frame, f'Total: {score}', (10,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        # cv.putText(frame, f'Eye: {eyeScore}', (10,200), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # cv.putText(frame, f'Mouth: {mouthScore}', (10,300), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        if(score > 0.4):
            flag+=1

            if(flag >= frame_check):
                mixer.music.play()
                cv.putText(frame, "******** STAY ALERT *********", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        else:
            flag = 0

        # new_frame_time = time.time()
        # fps = 1/(new_frame_time-prev_frame_time)
        # prev_frame_time = new_frame_time

        # fps = int(fps)
        # fps = str(fps)

        # cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # if cv.waitKey(1) & 0xFF == ord("q"):
            
    cap.release()

@app.route('/predict')
def predict():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=False)