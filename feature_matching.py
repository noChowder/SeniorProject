import numpy as py
import cv2 as cv
import matplotlib.pyplot as plt
# from iris_detection import iris_detection

path1 = "./MMU-Iris-Database/2/left/bryanl1.bmp"
path2 = "./MMU-Iris-Database/1/left/aeval3.bmp"

# id1 = iris_detection(path1)
# id2 = iris_detection(path2)

# id1.load_image()
# id1.convert_im2gray()
# id1.edge_detection()
# id1.get_pupil()
# id1.get_iris()
# id1.extract_iris()
# id1.crop_img()

# id2.load_image()
# id2.convert_im2gray()
# id2.edge_detection()
# id2.get_pupil()
# id2.get_iris()
# id2.extract_iris()
# id2.crop_img()

# img1 = cv.cvtColor(id1.work_img, cv.COLOR_GRAY2BGR)
# img2 = cv.cvtColor(id2.work_img, cv.COLOR_GRAY2BGR)

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()

print(len(matches))