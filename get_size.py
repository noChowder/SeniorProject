import numpy as py
import cv2 as cv

# directory = "./s_t_eyes/s1.bmp"
directory = "./MMU-Iris-Database/1/right/aevar3.bmp"

img = cv.imread(directory)

print(img.shape[0])
print(img.shape[1])