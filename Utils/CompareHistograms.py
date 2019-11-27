import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

image_path = 'databaseCroped/subject01-centerlight.jpg'

img = cv.imread(image_path, 0)
imgEqualised = cv.equalizeHist(img)

fig = plt.figure(dpi=150)
fig.suptitle('Comparação', fontsize=12)

ax1 = fig.add_axes([0.10, 0.60, 0.25, 0.25])
ax1.set_axis_off()
ax1.imshow(img, 'gray')
ax1.set_title('(a)', x=-0.2, y=0.4)

ax2 = fig.add_axes([0.10, 0.2, 0.25, 0.25])
ax2.set_axis_off()
ax2.imshow(imgEqualised, 'gray')
ax2.set_title('(b)', x=-0.2, y=0.4)

ax3 = fig.add_axes([0.4, 0.60, 0.5, 0.25])
dataLine = np.reshape(img, img.size)
ser1 = pd.Series(dataLine)
hist_img_orig = sns.distplot(ser1, 25, kde=False)
ax3.set_xlim((0, 255))
ax3.set_title('Histograma da imagem original', fontsize='10')

ax4 = fig.add_axes([0.4, 0.2, 0.5, 0.25])
dataLine2 = np.reshape(imgEqualised, imgEqualised.size)
ser2 = pd.Series(dataLine2)
hist_img_eq = sns.distplot(ser2, 25, kde=False)
ax4.set_xlim((0, 255))
ax4.set_title('Histograma da imagem equalizada', fontsize='10')
plt.show()
