import cv2 as cv
img = cv.imread("images\dog.jpg")
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("dog",grey)
cv.waitKey(0)