import os
import numpy as np
import cv2 as cv

DIR = r'D:\OpenCV\images\data'
IMAGE_PATH = r'D:\OpenCV\images\test.webp'   # <-- change to the image you want to test

# Same way your training script builds it, so the order matches
people = [x for x in os.listdir(DIR)]       # ['Ben', 'peter', 'tony']

haar = cv.CascadeClassifier(r'D:\OpenCV\haar_face.xml')

features = np.load(r'D:\OpenCV\features.npy', allow_pickle=True)
lables = np.load(r'D:\OpenCV\lables.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'D:\OpenCV\face_trained.yml')

img = cv.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    lable, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[lable]} with a confidence of {confidence}')

    cv.putText(img, str(people[lable]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)