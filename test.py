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

def test_leftEye_registeredEyes_left(subTestNum, subjectTest, subSampNum, subjectSamp, eyeNum):
    # test a subject left eye against a set of a subjects left eyes
    dirTest = "./MMU-Iris-Database/" + str(subTestNum) + "/left/" + subjectTest + "l" + str(eyeNum) + ".bmp"
    idTest = iris_detection(dirTest)
    idTest.detect()

    desSamp = []
    for i in range(1, 6):
        dirSamp = "./MMU-Iris-Database/" + str(subSampNum) + "/left/" + subjectSamp + "l" + str(i) + ".bmp"
        idSamp = iris_detection(dirSamp)
        idSamp.detect()
        desSamp.append(idSamp.des)
    
    total = 0
    for s in range(0, 5):
        bfmatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bfmatcher.match(idTest.des, desSamp[s])
        matches = sorted(matches, key = lambda x:x.distance)
        count = 0
        for e in matches:
            if (e.distance < 37):
                count += 1
        if (count > 10):
            total += 1
    
    if (total >= 3):
        print("\nEye is a match.\n")
    else:
        print("\nEye does not match.\n")

def main():
    # test_aeval()
    # test_pcll()
    # test_sitil_pscll()
    # test_leftEye_registeredEyes_left(2, "bryan", 5, 21, "mazwan")
    test_leftEye_registeredEyes_left(1, "aeva", 2, "bryan", 5)
    pass

if __name__ == "__main__":
    main()