import cv2
import numpy as np

imagee = cv2.imread('imagee.jpg', 0)
scale_percent = 50  # percent of original size
width = int(imagee.shape[1] * scale_percent / 100)
height = int(imagee.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image = cv2.resize(imagee, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', image.shape)
cv2.imshow("Resized image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


kernel = np.ones((2, 2), np.uint8)  # kernel for erosion
erosion = cv2.erode(image, kernel, iterations=1)  # increase black area
t = cv2.fastNlMeansDenoising(erosion, None, 20, 21, 7)  # remove some noise and bluring

######################implement skeleton fuction ###########################################
skeleton = np.zeros(t.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
size = np.size(t)
finish = False
returnn, t = cv2.threshold(t, 108, 255, cv2.THRESH_BINARY_INV)  # binary transform + filter
while (not finish):
    eroded = cv2.erode(t, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(t, temp)
    skeleton = cv2.bitwise_or(skeleton, temp)
    t = eroded.copy()
    zeros = size - cv2.countNonZero(t)
    if zeros == size:
        finish = True
kernel2 = np.ones((7, 7), np.uint8)
d_skeleton = cv2.dilate(skeleton, kernel2, iterations=1)
e_skeleton = cv2.erode(d_skeleton, kernel2, iterations=1)  # <- image after preprocessing
cv2.imshow("Skeleton", e_skeleton)
cv2.waitKey()
cv2.destroyAllWindows()
