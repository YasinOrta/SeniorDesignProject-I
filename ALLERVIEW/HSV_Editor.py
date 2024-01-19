import cv2
import numpy as np


def nothing(x):
    pass


# Load in image
# 0, 47, 128, 17, 109, 206
image_path = "images/3.jpeg"

original_img = cv2.imread(image_path)
resizeImage_var = original_img

scale_percent = 70
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv2.resize(resizeImage_var, dim, interpolation=cv2.INTER_AREA)

image = resizedImage

# Create a window
cv2.namedWindow('image')
cv2.namedWindow("hsvEditor")

# create trackbars for color change
cv2.createTrackbar('HMin', 'hsvEditor', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'hsvEditor', 0, 255, nothing)
cv2.createTrackbar('VMin', 'hsvEditor', 0, 255, nothing)
cv2.createTrackbar('HMax', 'hsvEditor', 0, 179, nothing)
cv2.createTrackbar('SMax', 'hsvEditor', 0, 255, nothing)
cv2.createTrackbar('VMax', 'hsvEditor', 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'hsvEditor', 179)
cv2.setTrackbarPos('SMax', 'hsvEditor', 255)
cv2.setTrackbarPos('VMax', 'hsvEditor', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while (1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'hsvEditor')
    sMin = cv2.getTrackbarPos('SMin', 'hsvEditor')
    vMin = cv2.getTrackbarPos('VMin', 'hsvEditor')

    hMax = cv2.getTrackbarPos('HMax', 'hsvEditor')
    sMax = cv2.getTrackbarPos('SMax', 'hsvEditor')
    vMax = cv2.getTrackbarPos('VMax', 'hsvEditor')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
            hMin, sMin, vMin, hMax, sMax, vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

"""
# unfinished code piece for applying hsv threshold 
image_path = "images/3.jpeg"

original_img = cv.imread(image_path)
resizeImage_var = original_img

scale_percent = 70
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)


result = resizedImage.copy()
image = cv.cvtColor(resizedImage, cv.COLOR_BGR2HSV)
lower = np.array([155,25,0])
upper = np.array([179,255,255])
mask = cv.inRange(image, lower, upper)
result = cv.bitwise_and(result, result, mask=mask)

cv.imshow('mask', mask)
cv.imshow('result', result)
cv.waitKey(0)
cv.destroyAllWindows()
"""
