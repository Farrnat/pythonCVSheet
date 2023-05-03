import argparse
import cv2
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="sh2.jpg", help="path to img")
args = vars(ap.parse_args())

# Read and display original image
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Apply grayscase color pallete and display image
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_gray)

# Apply threshold and display image
img_threshold = cv2.threshold(img_gray, 145, 255, cv2.THRESH_BINARY_INV)[1]
img_threshold = cv2.bitwise_not(img_threshold)
cv2.imshow("Thresh", img_threshold)

#Threshold erosion
kernel = np.ones((3, 3), np.uint8)
img_erosion = cv2.erode(img_threshold, kernel, iterations=3)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
cv2.imshow("ErodedDilatedImg", img_dilation)

#testVer erosion
#opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
#cv2.imshow("123", opening)


# Find contours and display image
img_contours = image.copy()
contours, hierarchy = cv2.findContours(
    img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
cnts = imutils.grab_contours((contours, hierarchy))
c = max(cnts, key=cv2.contourArea)
cv2.drawContours(img_contours, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv2.boundingRect(c)
text = "original, num_pts={}".format(len(c))
cv2.putText(
    img_contours, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
)

print("[INFO] {}".format(text))
cv2.imshow("Original Contour", img_contours)
cv2.waitKey(0)

for eps in np.linspace(0.001, 0.05, 10):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)
    output = image.copy()
    cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
    text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
    cv2.putText(
        output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    print("[INFO] {}".format(text))
    cv2.imshow("Approximated Contour", output)
    cv2.waitKey(0)