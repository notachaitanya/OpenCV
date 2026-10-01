import cv2 as cv

vid = cv.VideoCapture(0)

while True:
    ifTrue, frame = vid.read()

    haar = cv.CascadeClassifier("haar_face.xml")

    face = haar.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3)

    for(x,y,w,h) in face:
        cv.rectangle(frame,(x,y),((x+w),(y+h)),(0,255,0),thickness=2)
        cv.imshow("frame",frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()
vid.release()