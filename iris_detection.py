from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv

class iris_detection():
    def __init__(self, image_path):
        self.cimg = None # original color image
        self.gimg = None # working gray image
        self.img_path = image_path
        self.edges = None

    def load_image(self):
        self.cimg = cv.imread(self.img_path)

        if type(self.cimg) is type(None):
            return False
        else:
            return True

    def convert_im2gray(self):
        self.gimg = cv.cvtColor(self.cimg, cv.COLOR_BGR2GRAY)

    def edge_detection(self):
        self.gimg = cv.GaussianBlur(self.gimg, (7,7), cv.BORDER_DEFAULT)
        # self.img = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)
        lower_thres = 17 #17
        self.gimg = cv.Canny(self.gimg, lower_thres, 3 * lower_thres)

    def get_pupil(self):
        ret, self.gimg = cv.threshold(self.gimg, 50, 255, 0)
        # img, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(img, contours, -1, (0,255,0), 3)
        # im = cv.imread('test.bmp')
        # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # ret, thresh = cv.threshold(self.img, 127, 255, 0)
        # contours, hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(thresh, contours, -1, (0,255,0), 3)
        # Hough Transform
        circles = cv.HoughCircles(self.gimg, cv.HOUGH_GRADIENT, 1, 20, param1=300, param2=0.9, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(self.cimg, (i[0],i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv.circle(self.cimg, (i[0],i[1]), 2, (0,0,255), 3)
        cv.imshow('detected circles', self.cimg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            cv.imshow("result", self.gimg)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

for i in range(1):
    id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
    id.detect()
