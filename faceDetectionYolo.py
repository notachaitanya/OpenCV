import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np

DIR = r"D:\OpenCV\images\data\tony"
images = []
model = YOLO("yolov8m-face.pt")
face_rec = cv.face.LBPHFaceRecognizer_create()
faces =[]
lables =[]
if not images:
    for files in os.listdir(DIR):
        images.append(files)

for img in images:
    path = os.path.join(DIR,img)
    photo = cv.imread(path)

    image = model(photo)
    x1,y1,x2,y2 = image[0].boxes.xyxy[0].int().tolist()

    face = photo[y1:y2,x1:x2]
    gray = cv.cvtColor(face,cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray,(200,200))
    faces.append(gray)
    lables.append(0)
    
faces = np.array(faces)
lables = np.array(lables)

face_rec.train(faces,lables)
face_rec.write("yoloFaceRec.yml")
