from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv

class iris_detection():
    def __init__(self, image_path):
        self.orig_img = None    # original image
        self.work_img = None    # working image
        # self.polr_img = None    # image in polar coords
        self.pupil = None
        self.iris = None
        self.img_path = image_path
        self.edges = None

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
        circle = cv.HoughCircles(self.work_img, cv.HOUGH_GRADIENT, 1, 300, param1=80, param2=0.8, minRadius=0, maxRadius=35)
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
        cv.circle(mask, (self.iris[0], self.iris[1]), self.iris[2], (255,255,255), -1)
        cv.circle(mask, (self.pupil[0], self.pupil[1]), self.pupil[2], (0,0,0), -1)
        self.load_image()   # reset original image
        self.convert_im2gray()  # reset working image to grayscale of original image
        self.work_img = cv.bitwise_and(self.work_img, mask)
        # cv.imshow("mask", mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    
    def resize_img(self):
        # resize the image to show only the filtered parts
        X1 = self.iris[1] - self.iris[2]
        X2 = self.iris[1] + self.iris[2]
        Y1 = self.iris[0] - self.iris[2]
        Y2 = self.iris[0] + self.iris[2]
        self.work_img = self.work_img[X1:X2, Y1:Y2]

    def remove_extremities(self):
        # removes extreme pixels from image (eye-lashes)
        self.work_img = np.uint8(self.work_img > 40) * np.uint8(self.work_img < 140) * self.work_img

    def increase_contrast(self):
        # increase intensities of iris pixels
        M, N = self.work_img.shape
        for x in range(M):
            for y in range(N):
                if (self.work_img[y,x] == 0):
                    continue
                self.work_img[y,x] = ((np.double(self.work_img[y,x]) - 40) / (130-60)) * 255

    def normalize(self):
        # convert cartesian image to polar coords
        img = self.work_img.astype(np.float32)
        rows,cols = self.work_img.shape
        center = (cols/2, rows/2)
        radius = np.sqrt(rows**2 + cols**2)/2
        polar_img = cv.linearPolar(img, center, radius, cv.WARP_FILL_OUTLIERS)
        # self.work_img = polar_img.astype(np.uint8)
        polar_img = polar_img.astype(np.uint8)
        # rotate and resize image to compare with MATLAB
        M = cv.getRotationMatrix2D(((cols)/2, (rows)/2), 270, 1)
        polar_img = cv.warpAffine(polar_img, M, (cols, rows))
        self.work_img = cv.resize(polar_img, (self.work_img.shape[0]*4, self.work_img.shape[1]*2))
        # cv.imshow("Polar", polar_img)

    def extract_features(self):
        # feature detection using ORB
        orb = cv.ORB_create()
        key_points = orb.detect(self.work_img, None)
        key_points, des = orb.compute(self.work_img, key_points)
        self.work_img = cv.drawKeypoints(self.work_img, key_points, None, color=(0,255,0), flags=0)

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            self.get_iris()
            self.extract_iris()
            self.resize_img()
            self.remove_extremities()
            self.increase_contrast()
            self.normalize()
            self.extract_features()
            
            cv.imshow("Result", self.work_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

for i in range(12):
    id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
    print("Viewing eye number: \t" + str(i + 1) + "\n")
    id.detect()

# id = iris_detection("circle.bmp")
# id.detect()