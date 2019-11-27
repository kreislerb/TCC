import pandas as pd
import numpy as np
from Model.CreateDataset import CreateDataset
from sklearn.svm import SVC
from tqdm import tqdm
from time import time
from sklearn.model_selection import cross_val_score

# IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

database2 = pd.read_csv('../../PCA/CSV/PCA_ATT_DATABASE_90.csv', index_col='Unnamed: 0')
database2.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])

data2 = database2.T
labels2 = list()
for col in data2.columns:
    labels2.append(col)

def calcule_model_svm(kernel, n_c, database, labels):

    ds = CreateDataset('CSV/Time_process', "SVM_{}".format(kernel))

    c = np.logspace(-10, 10, n_c, base=10)


    if kernel == 'sigmoid' or kernel == 'rbf':

        ds.set_columns("C Gamma Accuracy Std Time_train".split())
        gamma = np.logspace(-10, 10, n_c, base=10)


        for i in tqdm(range(n_c)):
            for j in range(n_c):
                t0 = time()
                model = SVC(C=c[i], gamma=gamma[j], kernel=kernel, random_state=1)
                scores = cross_val_score(model, database, labels, cv=11, scoring='accuracy')
                # CRIA TABELA COM OS PARÂMETROS E ACURÁCIA DO MODELO
                ds.insert_row([model.C, model.gamma, scores.mean(), scores.std(), time() - t0])
                del model

    elif kernel == 'linear':
        del c
        c = np.logspace(-20, 20, n_c, base=10)
        ds.set_columns("C Accuracy Std Time_train".split())
        for i in tqdm(range(n_c)):
            t0 = time()
            model = SVC(C=c[i], kernel=kernel, random_state=1)
            scores = cross_val_score(model, database, labels, cv=10, scoring='accuracy')
            # CRIA TABELA COM OS PARÂMETROS E ACURÁCIA DO MODELO
            ds.insert_row([model.C, scores.mean(), scores.std(), time() - t0])
            del model
    else:
        ds.set_columns("C Gamma Degree Accuracy Std Time_train".split())
        gamma = np.logspace(-10, 10, n_c, base=10)

        for i in tqdm(range(n_c)):
            for j in range(n_c):
                for k in range(2, 6):
                    t0 = time()
                    model = SVC(C=c[i], gamma=gamma[j], degree=k, kernel=kernel, random_state=1)
                    scores = cross_val_score(model, database, labels, cv=11, scoring='accuracy')
                    # CRIA TABELA COM OS PARÂMETROS E ACURÁCIA DO MODELO
                    ds.insert_row([model.C, model.gamma, model.degree, scores.mean(), scores.std(), time() - t0])
                    del model
    ds.save()


calcule_model_svm('linear', 100, database, labels)

