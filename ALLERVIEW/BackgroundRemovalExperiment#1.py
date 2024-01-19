import cv2 as cv
import numpy as np
import os


def Id_Returner(path_name):
    # Retrieves image id from image path
    start_index = path_name.rfind("(") + 1

    last_index = path_name.rfind(')')

    image_id = path_name[start_index: last_index]

    return image_id


# setting the image location
image_path = 'BackgroundRemovalExperiment#1/1 (13).jpg'
image_id = Id_Returner(image_path)

# reading the image
original_img = cv.imread(image_path)
resizeImage_var = original_img

# resizing the image
scale_percent = 40
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)

# converting the bgr to gray
gray = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)

# threshold boundaries;
# threshold upper 250 lower 80 works for img no: 1-6
# threshold upper 250 lower 90 works for img no: 7-8
# threshold upper 250 lower 150-170 works for img no: 9-13

if 0 < int(image_id) < 7:
    lower = np.array([80, 80, 80])
    upper = np.array([255, 255, 255])

elif 7 <= int(image_id) <= 8:
    lower = np.array([90, 90, 90])
    upper = np.array([255, 255, 255])

elif 9 <= int(image_id) <= 13:
    lower = np.array([150, 150, 150])
    upper = np.array([255, 255, 255])

# creating mask
thresh = cv.inRange(resizedImage, lower, upper)

# applying morphology
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# invert morph image
mask = 255 - morph

# apply mask to image
result = cv.bitwise_and(resizedImage, resizedImage, mask=mask)

cv.imshow("original resized", resizedImage)
cv.imshow("thresh", thresh)
cv.imshow("mask", mask)
cv.imshow("result", result)

# Writing Results
path = "C:/Users/Yasin Orta/PycharmProjects/ImageProcessing3/ALLERVIEW/BackgroundRemovalExperiment#1/Results"
print(image_id)

# cv.imwrite(os.path.join(path, "threshold result of {0}.jpeg".format(image_id)), thresh)
# cv.imwrite(os.path.join(path, "mask of the {0}.jpeg".format(image_id)), mask)
# cv.imwrite(os.path.join(path, "applied mask result {0}.jpeg".format(image_id)), result)


cv.waitKey(0)
cv.destroyAllWindows()
