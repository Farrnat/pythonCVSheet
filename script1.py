import numpy as np
import argparse
import imutils
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='sh2.jpg', help="path to img")
args = vars(ap.parse_args())

image = cv.imread(args['image'])
cv.imshow('Image', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 160, 255, cv.THRESH_BINARY_INV) [1]
cv.imshow("Thresh", thresh)

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)

output = image.copy()
cv.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv.boundingRect(c)
text = "original, num_pts={}".format(len(c))
cv.putText(output, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

print("[INFO] {}".format(text))
cv.imshow("Original Contour", output)
cv.waitKey(0)

for eps in np.linspace(0.001, 0.05, 10):
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, eps * peri, True)
	output = image.copy()
	cv.drawContours(output, [approx], -1, (0, 255, 0), 3)
	text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
	cv.putText(output, text, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	print("[INFO] {}".format(text))
	cv.imshow("Approximated Contour", output)
	cv.waitKey(0)