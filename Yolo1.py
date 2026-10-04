import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
vid = cv.VideoCapture(0)

while True:
    ifTrue, frame = vid.read()

    result=model(frame)
    cv.imshow("img",result[0].plot())

    if cv.waitKey(20) & 0xff == ord('d'):
        break

vid.release()
cv.destroyAllWindows()