# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:41:34 2020

@author: ADMIN
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import cv2

model = load_model('traffic_recognition.h5')

img_dim = 30

class_labels = [
     'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing veh over 3.5 tons',
    'Right-of-way at intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Veh > 3.5 tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve left',
    'Dangerous curve right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End speed + passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End no passing veh > 3.5 tons'] 

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:

    ret, frame = cap.read()
    cv2.rectangle(frame, (100, 100), (250, 250), (255, 0, 255), 3)
    roi = frame[100:500, 100:500]
    img = cv2.resize(roi, (img_dim,img_dim))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    pred = np.argmax(model.predict(img))
    color = (0,0,255)

    cv2.putText(frame, class_labels[pred], (50,50), font, 1.0, color, 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()