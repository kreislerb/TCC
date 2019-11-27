import numpy as np
import pandas as pd
import cv2 as cv
import os
from tqdm import tqdm

info = dict(database_name='ATT Database Cropped and Equalised', autor='Kreisler Brenner', date='16/11/2019',
            description='Banco com 400 imagens de faces recortadas e histograma equalizado, cada imagem '
                        'está contida em uma coluna com 10752 linhas')

pathIn = '../Database/ATT Database/ATTDAtabaseCropped/'
pathOut = '../Database/CSV/'

files = os.listdir(pathIn)
listFaces = []
index = []

for filename in tqdm(files):
    img = cv.imread(pathIn + filename, 0)
    img_resized = np.reshape(img, img.size)
    listFaces.append(img_resized)

    '''
    rename = filename.split('-')
    i = rename[0][7:] + rename[1][:-4]
    '''
    rename = filename.split('_')
    i = rename[0][1:]
    print(i)
    #sr1 = pd.Series(img_resized, name=index)
    index.append(i)

#Criando o dataFrame
df = pd.DataFrame(listFaces, index)
print(df.head())
#criando o arquivo .csv do banco
df.to_csv(pathOut+info['database_name']+'.csv', index=True)
#criando o arquivo .txt com informações do banco
try:
    arq = open(pathOut+info['database_name']+' - read_me.txt', 'r+')
except FileNotFoundError:
    arq = open(pathOut+info['database_name']+' - read_me.txt', 'w+')
    arq.write('Database: ' + info['database_name']+'\n')
    arq.write('Description: ' + info['description']+'\n')
    arq.write('Autor: ' + info['autor']+'\n')
    arq.write('Date: ' + info['date'])
arq.close()
