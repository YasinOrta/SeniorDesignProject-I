import cv2 as cv
import numpy as np

image_path = 'images/2.jpeg'

original_img = cv.imread(image_path)
resizeImage_var = original_img

scale_percent = 50
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)

# setting and finding the adaptive threshold result
# block size and C variable are given randomly
adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 4)

kernel = np.ones((3,3), dtype="uint8")
morphed_image = cv.morphologyEx(adaptive_threshold, cv.MORPH_CLOSE, kernel)
result = morphed_image

# resized_adaptive_threshold_result in short ratr
ratr = cv.resize(result, dim, interpolation=cv.INTER_AREA)

# setting and finding contours
contours, _ = cv.findContours(ratr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_drawn = cv.drawContours(resizedImage, contours, -1, (0, 0, 255), 1)

cv.imshow("Background Removal with Adaptive Threshold", ratr)
cv.imshow("Original Image with contours drawn", contours_drawn)

cv.waitKey()
cv.destroyAllWindows()


