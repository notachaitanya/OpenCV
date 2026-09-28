import cv2 as cv

img = cv.imread("images/dog.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

threshold,thresh = cv.threshold(gray,thresh=100,maxval=255,type=cv.THRESH_BINARY)
cv.imshow("gray",thresh)
cv.waitKey(0)