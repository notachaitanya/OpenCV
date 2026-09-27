import cv2 as cv 
img = cv.imread("images\dog.jpg")
canny = cv.Canny(img,125,175)

cv.imshow("img",canny)
cv.waitKey(0)
#canny means it highlits only outlines of the image