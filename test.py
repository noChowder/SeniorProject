from iris_detection import *

def compare(test_des, descriptors):
    for s in range(len(descriptors)):
        print("__________________________________________________\n" + "\033[32m" + "\nTest #" + str(s+1) + ":" + "\033[0m")
        bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bfmatcher.match(test_des, descriptors[s])
        matches = sorted(matches, key = lambda x:x.distance)
        count = 0
        for e in matches:
            if (e.distance < 37):
                count += 1
        # print("\nThere were " + str(count) + " feature pairs in this test.")
        if (count > 10):
            # print("\nCorrectly matched with sample eye number:\t" + str(s+1))
            print("\nEyes matched.")
        else:
            print("\033[31m" + "\nNot a match." + "\033[0m")
        print("__________________________________________________\n\n")

def test_aeval():
    descriptors = []
    for i in range(1, 6):
        dir = "./MMU-Iris-Database/1/left/aeval" + str(i) + ".bmp"
        id = iris_detection(dir)
        id.detect()
        descriptors.append(id.des)
    compare(descriptors[4-1], descriptors)

def test_pcll():
    descriptors = []
    for i in range(1, 6):
        dir = "./MMU-Iris-Database/29/left/pcll" + str(i) + ".bmp"
        id = iris_detection(dir)
        id.detect()
        descriptors.append(id.des)
    compare(descriptors[5-1], descriptors)

def test_sitil_pscll():
    descriptors = []
    for i in range(1, 6):
        dir = "./MMU-Iris-Database/29/left/pcll" + str(i) + ".bmp"
        id = iris_detection(dir)
        id.detect()
        descriptors.append(id.des)
    for i in range(1, 6):
        dir = "./MMU-Iris-Database/34/left/sitil" + str(i) + ".bmp"
        id = iris_detection(dir)
        id.detect()
        descriptors.append(id.des)
    for i in range(5):
        print("\nTesting sitil eye:\t" + str(i+1) + "\tagainst pcll:\t6-10\n")
        compare(descriptors[i], descriptors[5:10])

def main():
    # test_aeval()
    # test_pcll()
    test_sitil_pscll()
    pass

if __name__ == "__main__":
    main()