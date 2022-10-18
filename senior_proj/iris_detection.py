from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv

class iris_detection():
    def __init__(self, image_path):
        self.img = None
        self.img_path = image_path
        self.edges = None

    def load_image(self):
        self.img = cv.imread(self.img_path)

        if type(self.img) is type(None):
            return False
        else:
            return True

    def convert_im2gray(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def edge_detection(self):
        self.img = cv.GaussianBlur(self.img, (7,7), cv.BORDER_DEFAULT)
        # self.img = cv.adaptiveThreshold(self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)
        lower_thres = 17
        self.img = cv.Canny(self.img, lower_thres, 3 * lower_thres)

    def get_pupil(self):
        ret, self.img = cv.threshold(self.img, 50, 255, THRESH_BINARY)

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            cv.imshow("result", self.img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

for i in range(1):
    id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
    id.detect()
