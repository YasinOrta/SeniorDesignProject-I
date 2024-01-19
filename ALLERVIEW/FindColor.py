import cv2 as cv
import numpy as np

# Reading image
image_path = "images/3.jpeg"

original_img = cv.imread(image_path)
resizeImage_var = original_img

scale_percent = 70
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)
# -------
image = resizedImage
# Load image and HSV color threshold
original = image.copy()
image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
lower = np.array([0, 40, 140], dtype="uint8")
upper = np.array([15, 100, 210], dtype="uint8")
mask = cv.inRange(image, lower, upper)
detected = cv.bitwise_and(original, original, mask=mask)

# Remove noise
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

# Find contours and find total area
cnts = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area = 0
for c in cnts:
    area += cv.contourArea(c)
    cv.drawContours(original,[c], 0, (0,0,0), 2)

print(area)
cv.imshow("original", resizedImage)
cv.imshow('edited original', original)
cv.imshow('detected', detected)

# -------

# cv.imshow("result", resizedImage)

cv.waitKey(0)
cv.destroyAllWindows()
