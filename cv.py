import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import sort

def get_answer(contour):
        (x,y,w,h) = cv.boundingRect(contour)
        if w>=30 and h>=15 and w<=40 and h<=20:
            return True
        return False


path = f'tracnghiem.jpg'

img = cv.imread(path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#blurred = cv.GaussianBlur(gray, (7,7), 0)

thresh =cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 651, 19)

contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)



contours = filter(get_answer, contours)

contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

img_sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=1)
img_sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=1)
# print(img_sobelx, type(img_sobelx))
# cv.imshow('img_sobelx', img_sobelx)
# cv.imshow('img_sobely', img_sobely)
img_sobel = (img_sobelx + img_sobely)/2

for i in range(img_sobel.shape[0]):
    for j in range(img_sobel.shape[1]):
        if img_sobel[i][j] < 30:
            img_sobel[i][j] = 0
        else:
            img_sobel[i][j] = 255

# cv.imshow('sobel', img_sobel)

black = np.zeros(gray.shape[:2])

for i in range(len(contours)-150,len(contours)-2):
    cv.drawContours(black, contours, i, 255,1)
# cv.drawContours(black, contours, -1, 255, 1)

cv.imshow('imgg', black)
cv.imshow('img', thresh)

cv.waitKey(0)