import cv2 as cv
import os

pathIn = '../Database/ATT Database/ATTDatabaseCropped/'
pathOut = '../Database/ATT Database/ATTDatabaseEqualised/'
files = os.listdir(pathIn)
i = 1
for filename in files:

        print(filename)
        img = cv.imread(pathIn + filename, 0)
        imgEqualised = cv.equalizeHist(img)
        cv.imwrite(pathOut + filename, imgEqualised)

