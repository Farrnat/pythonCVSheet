import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('sh2.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

_, binary = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
plt.imshow(binary, cmap="gray")
plt.show()

contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
image = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
plt.imshow(img)
plt.show()















