import cv2 as cv

vid = cv.VideoCapture(0)
while True:
    ifTrue, frame = vid.read()
    
    cv.rectangle(frame,(0,0),(100,100),(0,255,0),thickness=2)
    cv.imshow("vid",frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
vid.release()
cv.destroyAllWindows()