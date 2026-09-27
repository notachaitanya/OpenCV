import cv2 as cv

vid = cv.VideoCapture(0)
while True:
    ifTrue, frame = vid.read()
    cv.imshow("vid",frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()