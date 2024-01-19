import cv2 as cv
import numpy as np

image_path = "images/3.jpeg"

original_img = cv.imread(image_path)
resizeImage_var = original_img

scale_percent = 70
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)

hsv_img = cv.cvtColor(resizedImage, cv.COLOR_BGR2HSV)

lower = np.array([0, 40, 140], dtype="uint8")
upper = np.array([15, 100, 210], dtype="uint8")

mask = cv.inRange(hsv_img, lower, upper)

result = cv.bitwise_and(resizedImage, resizedImage, mask=mask)

cv.imshow("result", result)
cv.imshow("original", resizedImage)

cv.waitKey(0)
cv.destroyAllWindows()
