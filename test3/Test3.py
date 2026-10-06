import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")
DIR = r"D:\OpenCV\images\data\tony"
face_Recognizer = cv.face.LBPHFaceRecognizer_create()
images=[]
faces = []
lables =[]

if not images:
    for img in os.listdir(DIR):
        images.append(img)


for img in images:
    path = os.path.join(DIR,img)

    pic = cv.imread(path)

    result = model(pic)
    x1,y1,x2,y2 = result[0].boxes.xyxy[0].int().tolist()
    pic = pic[y1:y2,x1:x2]
    grey_pic = cv.cvtColor(pic,cv.COLOR_BGR2GRAY)
    grey_pic = cv.resize(grey_pic,(200,200))
    faces.append(grey_pic)
    lables.append(0)

faces = np.array(faces)
lables = np.array(lables)

face_Recognizer.train(faces,lables)
face_Recognizer.write("Test3.yml")


