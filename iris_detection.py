from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog

class iris_detection():
    def __init__(self, image_path):
        self.orig_img = None    # original image
        self.work_img = None    # working image
        # self.polr_img = None    # image in polar coords
        self.pupil = None
        self.iris = None
        self.img_path = image_path
        self.edges = None
        self.kp = None
        self.des = None

    def load_image(self):
        self.orig_img = cv.imread(self.img_path)

        if type(self.orig_img) is type(None):
            return False
        else:
            return True

    def convert_im2gray(self):
        # convert to grayscale image
        self.work_img = cv.cvtColor(self.orig_img, cv.COLOR_BGR2GRAY)

    def edge_detection(self):
        # blur image with 7x7 window
        self.work_img = cv.GaussianBlur(self.work_img, (7,7), cv.BORDER_DEFAULT)

        # canny edge detection with lthres=30 uthresh=80
        self.work_img = cv.Canny(self.work_img, 30, 80)

    def get_pupil(self):
        # gets pupil circle
        # param2=0.8 allows for smaller circle detection
        circle = cv.HoughCircles(self.work_img, cv.HOUGH_GRADIENT, 1, 300, param1=80, param2=0.8, minRadius=17, maxRadius=35)
        circle = np.uint16(np.around(circle))
        for i in circle[0,:]:
            # draw the outer circle
            cv.circle(self.orig_img, (i[0],i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            # cv.circle(self.orig_img, (i[0],i[1]), 2, (0,0,255), 2)
            self.pupil = (i[0], i[1], i[2])

    def get_iris(self):
        # gets iris circle
        # param2=0.85 allows for larger circle detection
        circle = cv.HoughCircles(self.work_img, cv.HOUGH_GRADIENT, 1, 300, param1=80, param2=0.85, minRadius=35, maxRadius=91)
        circle = np.uint16(np.around(circle))
        for i in circle[0,:]:
            if( abs(self.pupil[0] - int(i[0])) > 11 or abs(self.pupil[1] - int(i[1])) > 11):    # sets iris size to 2*pupil if center is too far
                i[0] = self.pupil[0]
                i[1] = self.pupil[1]
                i[2] = 2.5 *self.pupil[2]
                cv.circle(self.orig_img, (i[0],i[1]), i[2], (0,255,0), 2)
                # cv.circle(self.orig_img, (i[0],i[1]), 2, (0,0,255), 2)
            else:
                if( int(i[2]) > 2.5*self.pupil[2] ):    # sets iris size to 2.5*iris size if detected size is greater than 2.5*iris size
                    i[2] = 2.5*self.pupil[2]
                # draw the outer circle
                cv.circle(self.orig_img, (i[0],i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                # cv.circle(self.orig_img, (i[0],i[1]), 2, (0,0,255), 2)
        self.iris = (i[0], i[1], i[2])

    def extract_iris(self):
        # filter pixels outside iris and inside pupil
        mask = np.zeros((self.orig_img.shape[0], self.orig_img.shape[1], 1), np.uint8)
        # iris mask
        cv.circle(mask, (self.iris[0], self.iris[1]), self.iris[2], (255,255,255), -1)
        # pupil mask
        cv.circle(mask, (self.pupil[0], self.pupil[1]), self.pupil[2], (0,0,0), -1)
        self.load_image()   # reset original image
        self.convert_im2gray()  # reset working image to grayscale of original image
        self.work_img = cv.bitwise_and(self.work_img, mask)
        # cv.imshow("mask", mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    
    def crop_img(self):
        # crop the image to show only the filtered parts
        X1 = self.iris[1] - self.iris[2]
        X2 = self.iris[1] + self.iris[2]
        Y1 = self.iris[0] - self.iris[2]
        Y2 = self.iris[0] + self.iris[2]
        self.work_img = self.work_img[X1:X2, Y1:Y2]

    def remove_extremities(self):
        # removes extreme pixels from image (eye-lashes)
        self.work_img = np.uint8(self.work_img > 50) * np.uint8(self.work_img < 130) * self.work_img

    def increase_contrast(self):
        # increase intensities of iris pixels
        M, N = self.work_img.shape
        for x in range(M):
            for y in range(N):
                if (self.work_img[y,x] == 0):
                    continue
                self.work_img[y,x] = ((np.double(self.work_img[y,x]) - 50) / (130-50)) * 255

    def normalize(self):
        # convert image to polar
        rows, cols = self.work_img.shape
        self.work_img = cv.warpPolar(self.work_img, (0,0), (cols/2, rows/2), 91, cv.WARP_FILL_OUTLIERS)

    def extract_features(self):
        # feature detection using ORB
        orb = cv.ORB_create()
        key_points = orb.detect(self.work_img, None)
        key_points, des = orb.compute(self.work_img, key_points)
        self.kp = key_points
        self.des = des
        self.work_img = cv.drawKeypoints(self.work_img, key_points, None, color=(0,255,0), flags=0)

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            self.get_iris()

            # cv.imshow("Eye", self.orig_img)

            self.extract_iris()
            self.crop_img()
            self.remove_extremities()
            self.increase_contrast()
            self.normalize()
            self.extract_features()

            # cv.imshow("Result", self.work_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # show image in matplot
            # img = self.work_img
            # img = cv.cvtColor(self.work_img, cv.COLOR_GRAY2BGR)
            # plt.imshow(img, cmap="gray"), plt.show()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

# num_of_eyes = 5
# subject_num = "2"
# subject_name = "bryan"

# left eye
# for i in range(num_of_eyes):
#     directory = "./MMU-Iris-Database/" + subject_num + "/left/" + subject_name + "l" + str(i+1) + ".bmp"
#     # id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
#     id = iris_detection(directory)
#     # print("Viewing left eye number: \t" + str(i + 1) + "\n")
#     id.detect()
#     stored_kp.append(id.kp)
#     stored_des.append(id.des)

# right eye
# for i in range(num_of_eyes):
#     directory = "./MMU-Iris-Database/" + subject_num + "/right/" + subject_name + "r" + str(i+1) + ".bmp"
#     # id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
#     id = iris_detection(directory)
#     print("Viewing right eye number: \t" + str(i + 1) + "\n")
#     id.detect()

# dir1 = "./MMU-Iris-Database/1/left/aeval1.bmp"
# dir2 = "./MMU-Iris-Database/1/left/aeval5.bmp"

def feature_match():
    # feature matching
    # dir1 = "./MMU-Iris-Database/1/left/aeval2.bmp"
    # dir2 = "./MMU-Iris-Database/2/left/bryanl2.bmp"
    dir1 = "./s_t_eyes/s10.bmp"
    dir2 = "./s_t_eyes/t1_10.bmp"

    id1 = iris_detection(dir1)
    id1.detect()

    id2 = iris_detection(dir2)
    id2.detect()

    bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bfmatcher.match(id1.des, id2.des)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv.drawMatches(id1.work_img, id1.kp, id2.work_img, id2.kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

    # change to modifer accuracy
    dist = 40
    count = 0
    for e in matches:
        if (e.distance < dist):
            count += 1
    print("\nThere are " + str(count) + " pairs < " + str(dist) + "\n")

def test_12():
    # test eyes against samples
    total = 0
    for e in range(1, 13):
        key_points = []
        descriptors = []


        test_eye = e

        test_img = "./s_t_eyes/t1_" + str(test_eye) + ".bmp"
        id = iris_detection(test_img)
        id.detect()
        key_points.append(id.kp)
        descriptors.append(id.des)

        for i in range(1, 13):
            samp_img = "./s_t_eyes/s" + str(i) + ".bmp"
            id = iris_detection(samp_img)
            id.detect()
            key_points.append(id.kp)
            descriptors.append(id.des)

            bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bfmatcher.match(descriptors[0], descriptors[i])
            matches = sorted(matches, key = lambda x:x.distance)

            count = 0
            for e in matches:
                if (e.distance < 37):
                    count += 1
            # print("\nThere were " + str(count) + " matches in this test.\n")
            if (count > 10):
                if (test_eye == i):
                    print("\nMatched test eye\t" + str(test_eye) + "\twith sample eye\t\t" + str(i) + "\033[32m" + "\tMATCH\n" + "\033[0m")
                    total += 1
                else:
                    print("\nMatched test eye\t" + str(test_eye) + "\twith sample eye\t\t" + str(i) + "\n")

    print("\033[32m" + "\nTOTAL CORRECT MATCHES: " + str(total) + "\n" + "\033[0m")

    pass

def main():
    # feature_match()
    test_12()
    pass

if __name__ == "__main__":
    main()