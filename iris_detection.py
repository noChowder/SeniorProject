from cv2 import THRESH_BINARY
import numpy as np
import cv2 as cv

class iris_detection():
    def __init__(self, image_path):
        self.cimg = None # original color image
        self.gimg = None # working gray image
        self.pupil = None
        self.img_path = image_path
        self.edges = None

    def load_image(self):
        self.cimg = cv.imread(self.img_path)

        if type(self.cimg) is type(None):
            return False
        else:
            return True

    def convert_im2gray(self):
        # convert to grayscale image
        self.gimg = cv.cvtColor(self.cimg, cv.COLOR_BGR2GRAY)

    def edge_detection(self):
        # blur image with 7x7 window
        self.gimg = cv.GaussianBlur(self.gimg, (7,7), cv.BORDER_DEFAULT)

        # canny edge detection with lthres=30 uthresh=80
        self.gimg = cv.Canny(self.gimg, 30, 80)

    def get_pupil(self):
        # gets pupil circle
        # param2=0.8 allows for smaller circle detection
        circles = cv.HoughCircles(self.gimg, cv.HOUGH_GRADIENT, 1, 300, param1=80, param2=0.8, minRadius=0, maxRadius=35)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(self.cimg, (i[0],i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv.circle(self.cimg, (i[0],i[1]), 2, (0,0,255), 2)
            self.pupil = (i[0], i[1], i[2])

    def get_iris(self):
        # gets iris circle
        # param2=0.85 allows for larger circle detection
        circles = cv.HoughCircles(self.gimg, cv.HOUGH_GRADIENT, 1, 300, param1=80, param2=0.85, minRadius=35, maxRadius=70)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            if( abs(self.pupil[0] - int(i[0])) > 10 or abs(self.pupil[1] - int(i[1])) > 10): # sets iris size to 2.5*pupil if center is too far
                cv.circle(self.cimg, (self.pupil[0], self.pupil[1]), 2.5*self.pupil[2], (0,255,0), 2)
                cv.circle(self.cimg, (self.pupil[0], self.pupil[1]), 2, (0,0,255), 2)
            else:
                if( int(i[2]) > 2.5*self.pupil[2] ): # sets iris size to 2.5*iris size if detected size is greater than 2.5*iris size
                    i[2] = 2.5*self.pupil[2]
                # draw the outer circle
                cv.circle(self.cimg, (i[0],i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                cv.circle(self.cimg, (i[0],i[1]), 2, (0,0,255), 2)

    def detect(self):
        if (self.load_image()):
            self.convert_im2gray()
            self.edge_detection()
            self.get_pupil()
            self.get_iris()
            cv.imshow("result", self.cimg)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print ('Image "' + self.img_path + '" could not be loaded.')

for i in range(12):
    id = iris_detection("./s_t_eyes/" + 's' + str(i + 1) + ".bmp")
    print("Viewing eye number: \t" + str(i + 1) + "\n")
    id.detect()
