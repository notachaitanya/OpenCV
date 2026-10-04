import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from PIL import Image

DIR = r"D:\OpenCV\images\data\tony"
images = []
model = YOLO("yolov8m-face.pt")

if not images:
    for files in os.listdir(DIR):
        images.append(files)

for img in images:
    path = os.path.join(DIR,img)
    photo = cv.imread(path)

image = model(photo)
cv.imshow("Tony",image[0].plot())
cv.waitKey(0)
cv.destroyAllWindows()