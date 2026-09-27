import cv2 as cv 
img = cv.imread("images/dog.jpg")

blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT) #more the number in that more will be teh blur, and those number should be odd
cv.imshow("img",blur)
cv.waitKey(0)