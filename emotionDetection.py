import cv2 as cv
import sys, os
import numpy as np

from keras.models import model_from_json
from keras.preprocessing import image

#Loading the model
model = model_from_json(open("fer.json", "r").read())
#Loading the weights
model.load_weights('fer.h5') 

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

if not cap.isOpened():
	print ("Camera not available!")
	exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print ("Frame not captured!")
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame_gray, 1.32, 5)
    
    for (x,y,w,h) in face:
        cv.rectangle(frame, (x,y) , (x+w, y+h), (255, 0, 0), 2)
        roi = frame_gray[y : y+h, x : x+w]
        roi = cv.resize(roi, (48,48))
        roi_pixels = image.img_to_array(roi)
        roi_pixels = np.expand_dims(roi_pixels, 0)
        #Normalising image
        roi_pixels/=255
        predictions = model.predict(roi_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')
        predicted_emotion = emotions[max_index]
        cv.putText(frame, predicted_emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_frame = cv.resize(frame, (1000, 700))
    cv.imshow('Emotion Detection', resized_frame) 

    if cv.waitKey(10) == ord('q'):
        break

cv.waitKey(0) 
cap.release()
cv.destroyAllWindows()

