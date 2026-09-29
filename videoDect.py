import cv2 as cv

vid = cv.VideoCapture(0)

while True:
    ifTrue, frame = vid.read()

    haar = cv.CascadeClassifier("haar_face.xml")

    face = haar.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=1)

    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),thickness=2,color=(0,255,0))
    cv.imshow("feed",frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()
cv.release(vid)