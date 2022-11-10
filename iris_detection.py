from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv

class iris_detection():
    def __init__(self, image_path):
        self.orig_img = None # original image
        self.work_img = None # working image
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
            if( abs(self.pupil[0] - int(i[0])) > 11 or abs(self.pupil[1] - int(i[1])) > 11): # sets iris size to 2*pupil if center is too far
                i[0] = self.pupil[0]
                i[1] = self.pupil[1]
                i[2] = 2.5 *self.pupil[2]
                cv.circle(self.orig_img, (i[0],i[1]), i[2], (0,255,0), 2)
                # cv.circle(self.orig_img, (i[0],i[1]), 2, (0,0,255), 2)
            else:
                if( int(i[2]) > 2.5*self.pupil[2] ): # sets iris size to 2.5*iris size if detected size is greater than 2.5*iris size
                    i[2] = 2.5*self.pupil[2]
                # draw the outer circle
                cv.circle(self.orig_img, (i[0],i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                # cv.circle(self.orig_img, (i[0],i[1]), 2, (0,0,255), 2)
        self.iris = (i[0], i[1], i[2])

    def extract_iris(self):
        # filter pixels outside iris and inside pupil
        pass

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            self.get_iris()
            cv.imshow("result", self.orig_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

for i in range(12):
    id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
    print("Viewing eye number: \t" + str(i + 1) + "\n")
    id.detect()
