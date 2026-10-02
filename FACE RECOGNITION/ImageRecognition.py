import cv2 as cv
import numpy as np
import os

people = []

for x in os.listdir(r'D:\OpenCV\images\data'):
    people.append(x)

DIR = r'D:\OpenCV\images\data'
haar = cv.CascadeClassifier("haar_face.xml")
features = []
lables = []

def train():
    for person in people:
        path = os.path.join(DIR,person)
        lable = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            Img = cv.imread(img_path)
            gray = cv.cvtColor(Img,cv.COLOR_BGR2GRAY)

            faces = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces:
                faces_roi = gray[y:y+h , x:x+w]
                features.append(faces_roi)
                lables.append(lable)

train()
features = np.array(features, dtype='object')
lables = np.array(lables)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,lables)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('lables.npy',lables)
