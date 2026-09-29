import cv2 as cv
img = cv.imread("images/tony.jpg")
grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("tony",grey)

#reading the haar xml file
haar = cv.CascadeClassifier("haar_face.xml")

face = haar.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in face:
    cv.rectangle(img,(x,y),(x+w,y+h),thickness=2,color=(0,255,0))

cv.imshow("tony",img)

cv.waitKey(0)