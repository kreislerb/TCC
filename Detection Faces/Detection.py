import cv2 as cv
import os
from tqdm import tqdm



face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
pathIn = '../Database/ATT Database/ATTDatabaseEqualised/'
pathOut = '../Database/ATT Database/ATTDatabaseCropped/'
files = os.listdir(pathIn)

# Read the model
#model = cv.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')


for filename in tqdm(files):

    img = cv.imread(pathIn + filename)

    faces = face_cascade.detectMultiScale(img, 1.5, 5)
    for (x, y, w, h) in faces:

        offset = 0
        width = x + w - offset
        height = y + h - offset
        y1 = int(y + offset / 2)
        x1 = int(x + offset / 2)
        # cv.rectangle(img, (x1, y1), (width, height), (255, 0, 0), 1)
        roi_color = img[y1:height, x1:width]

        # Redimensiona a face extraída usando a interpolação linear
        #imgOut = cv.resize(roi_color, (150,150))
        imgOut = cv.resize(roi_color, (96,112))

        # Armazena imagem no local especificado
        cv.imwrite(pathOut + filename, imgOut)
