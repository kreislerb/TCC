import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


info = dict(name='PCA_ATT_DATABASE', autor='Kreisler Brenner', date='16/11/2019',
            description='Redução de dimensionalidade das imagens do ATT DATABASE com o calculo do PCA')

pathOut = 'CSV/'
# importando database
database = pd.read_csv('../Database/CSV/ATT Database Cropped and Equalised.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

def processPCA(var):

    print(database.head())
    # Normalizando banco de dados
    scaler = StandardScaler()
    scaler.fit(database)
    scaled_data = scaler.transform(database)

    # Calcula o PCA com "var" de variancia
    pca = PCA(var/100)
    pca.fit(scaled_data)
    print('Numero de Componentes: ' + str(pca.n_components_))
    xpca = pca.transform(scaled_data)
    df = pd.DataFrame(xpca, index=database.index)
    print(df.head())
    # criando o arquivo .csv do banco
    df.to_csv(pathOut+info['name']+"_{}".format(var)+'.csv', index=True)

def curve_pca_num_comp():

    # Normalizando banco de dados
    scaler = StandardScaler()
    scaler.fit(database)
    scaled_data = scaler.transform(database)
    num_comp = list()

    for va in tqdm(np.arange(50,100)):

        # Calcula o PCA com "var" de variancia
        pca = PCA(va/100)
        pca.fit(scaled_data)
        num_comp.append(pca.n_components_)
        del pca

    ds = pd.DataFrame(np.asarray(num_comp),np.arange(50,100),['num_comp'])
    ds.to_csv('NumComp.csv')


def plot_curve_num_comp():
    curve = pd.read_csv('NumComp.csv', index_col='Unnamed: 0')
    plt.plot(curve['num_comp'], curve.index)
    plt.plot(45,90,marker='x', markersize=15)
    plt.ylabel('Variância (%)')
    plt.xlabel('Número de componentes')
    plt.margins(x=0)
    plt.show()


def processPCA2(var):

    print(database.head())
    # Normalizando banco de dados
    scaler = StandardScaler()
    scaler.fit(database)
    scaled_data = scaler.transform(database)

    # Calcula o PCA com "var" de variancia
    pca = PCA(var/100)
    pca.fit(scaled_data)
    print('Numero de Componentes: ' + str(pca.n_components_))
    print(pca.components_.shape)
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))

processPCA(90)
#plot_curve_num_comp()