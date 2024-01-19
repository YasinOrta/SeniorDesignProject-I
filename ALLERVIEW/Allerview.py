import cv2 as cv


def Id_Returner(path_name):

    # Retrieves image id from image path
    start_index = path_name.rfind("/") + 1

    last_index = path_name.rfind('.')

    image_id = path_name[start_index: last_index]

    return image_id


image_path = "images/3.jpeg"

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
adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)
result = adaptive_threshold

# resized_adaptive_threshold_result in short ratr
ratr = cv.resize(result, dim, interpolation=cv.INTER_AREA)

# setting and finding contours
contours, _ = cv.findContours(ratr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_drawn = cv.drawContours(resizedImage, contours, -1, (255, 0, 255), 1)

cv.imshow("Background Removal with Adaptive Threshold", ratr)
cv.imshow("Original Image with contours drawn", contours_drawn)

# saving the results, deleting # sign will make program to write the output files
path = "C:/Users/Yasin Orta/PycharmProjects/ImageProcessing3/AllerviewData/adaptive_threshold_results + contours drawn"
image_id = Id_Returner(image_path)

# cv.imwrite(os.path.join(path,"contours drawn to image{0}.jpeg".format(image_id)),contours_drawn)
# cv.imwrite(os.path.join(path,"adapted threshold applied image{0}.jpeg".format(image_id)),ratr)
# cv.imwrite(os.path.join(path,"original image{0}.jpeg".format(image_id)),original_img)

cv.waitKey()
cv.destroyAllWindows()


