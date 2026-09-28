import cv2 as cv
import numpy as np
blank = np.zeros((400,400,3),dtype="uint8")
r=cv.rectangle(blank.copy(),(30,30),(370,370),(0,255,0),-1)
c=cv.circle(blank.copy(),(200,200),200,(0,255,0),-1)

#AND operator
And = cv.bitwise_and(r,c)
cv.imshow("AND",And)

#OR operator
Or = cv.bitwise_or(r,c)
cv.imshow("OR",Or)

#NOT operator
Not=cv.bitwise_not(c)
cv.imshow("NOT",Not)

#XOR operator
Xor = cv.bitwise_xor(r,c)
cv.imshow("XOR",Xor)

cv.waitKey(0)