import numpy as np
import cv2
from copy import deepcopy
from matplotlib import pyplot as plt

img = cv2.imread('barcode_10.PNG') # 입력으로 이미지를 넣어준다.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gradX = cv2.Sobel(gray, cv2.CV_8UC1, 1, 0, ksize = 3)

gradY = cv2.Sobel(gray, cv2.CV_8UC1, 0, 1, ksize = 3)

subtract = cv2.subtract(gradX,gradY)

blur = cv2.GaussianBlur(subtract, (9,9), 0)

th, thresh= cv2.threshold(blur,50,255,cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

eroded = cv2.erode(closed, kernel)

eroded = cv2.erode(eroded, kernel)

dilated = cv2.dilate(eroded, kernel)

dilated = cv2.dilate(dilated, kernel)

(_, cnts, _) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

rect = cv2.minAreaRect(c)

box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(img, [box], 0, (0,255,0), 2)


titles = ['Gray Image', 'gradX' , 'gradY', 'gradX - gradY', 'blur image','threshold image', 'closed image' , 'eroded and dilated image']

images = [gray, gradX , gradY, subtract, blur, thresh, closed, eroded]

for i in range(8):
    plt.subplot(3,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv2.imshow("result image", img)

cv2.waitKey(0) 
