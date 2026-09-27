import cv2 as cv
vid = cv.VideoCapture(0)
while True:
    ifTrue ,frame = vid.read()
    canny = cv.Canny(frame,125,300)
    cv.imshow("feed",canny)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

vid.release()
cv.destroyAllWindows()