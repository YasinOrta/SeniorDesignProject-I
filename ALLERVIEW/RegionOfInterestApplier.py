import cv2 as cv
import numpy as np


def Id_Returner(path_name):

    # Retrieves image id from image path
    start_index = path_name.rfind("/") + 1

    last_index = path_name.rfind('.')

    image_id = path_name[start_index: last_index]

    return image_id


image_path = "images/3.jpeg"

original_img = cv.imread(image_path)
resizeImage_var = original_img

scale_percent = 70
width = int(resizeImage_var.shape[1] * scale_percent / 100)
height = int(resizeImage_var.shape[0] * scale_percent / 100)
dim = (width, height)
resizedImage = cv.resize(resizeImage_var, dim, interpolation=cv.INTER_AREA)

# region of interest -> roi coordinates input
# select a roi and then press space or enter button, cancel the selection by presscing c button
roi = cv.selectROI("select the region of interest", resizedImage)
# cropping the image
cropped_image = resizedImage[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# cv.imshow("original", original_img)
cv.imshow("resized image", resizedImage)
cv.imshow("cropped image", cropped_image)

"""
# saving the results, deleting # sign will make program to write the output files
path = "C:/Users/Yasin Orta/PycharmProjects/ImageProcessing3/AllerviewData/adaptive_threshold_results + contours drawn"
image_id = Id_Returner(image_path)

# cv.imwrite(os.path.join(path,"contours drawn to image{0}.jpeg".format(image_id)),contours_drawn)
# cv.imwrite(os.path.join(path,"adapted threshold applied image{0}.jpeg".format(image_id)),ratr)
# cv.imwrite(os.path.join(path,"original image{0}.jpeg".format(image_id)),original_img)
"""

cv.waitKey()
cv.destroyAllWindows()


