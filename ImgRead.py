import cv2 as cv

img = cv.imread("images\dog.jpg")
cv.imshow("dog",img)

cv.waitKey(0)