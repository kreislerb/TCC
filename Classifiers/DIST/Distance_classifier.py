import pandas as pd
import numpy as np
import sklearn.neighbors as ng
from sklearn.neighbors import DistanceMetric
from time import time

info = dict(name='Distance_Classifier_time', autor='Kreisler Brenner', date='31/10/2019',
            description='Treinamento e classificação utilizando metricas de similaridade')

pathOut = ''
# importando database
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

# INICIALIZANDO CLASSIFICADORES DE METRICAS
euclidean = ng.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
manhattan = ng.KNeighborsClassifier(n_neighbors=1, metric='manhattan')
chebyshev = ng.KNeighborsClassifier(n_neighbors=1, metric='chebyshev')

DistanceMetric.get_metric('mahalanobis', V=np.cov(database))
mahalanobis = ng.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='mahalanobis',
                                      metric_params={'V': np.cov(database)})

# PREPARANDO VETOR DE LABELS
# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])

# RESULTADOS

from sklearn.model_selection import cross_val_score


# VALIDAÇÃO CRUZADA UTILIZANDO METRICA EUCLIDIANA
t0 = time()
scores = cross_val_score(euclidean, database, labels, cv=11, scoring='accuracy')
time1 = time()-t0
print("Accuracy EUCLIDIANA: %0.2f (+/- %0.2f)   Time: %0.5f" % (scores.mean(), scores.std(), time1))
# VALIDAÇÃO CRUZADA UTILIZANDO METRICA MANHATTAN
t0 = time()
scores2 = cross_val_score(manhattan, database, labels, cv=11, scoring='accuracy')
time2 = time()-t0
print("Accuracy MANHATTAN: %0.2f (+/- %0.2f)    Time: %0.5f" % (scores2.mean(), scores2.std(), time2))
# VALIDAÇÃO CRUZADA UTILIZANDO METRICA CHEBYSHEV
t0 = time()
scores3 = cross_val_score(chebyshev, database, labels, cv=11, scoring='accuracy')
time3 = time()-t0
print("Accuracy CHEBYSHEV: %0.2f (+/- %0.2f)    Time: %0.5f" % (scores3.mean(), scores3.std(), time3))
# VALIDAÇÃO CRUZADA UTILIZANDO METRICA MAHALANOBIS
t0 = time()

scores4 = cross_val_score(mahalanobis, database, labels, cv=11, scoring='accuracy')
time4 = time()-t0
print("Accuracy MAHALANOBIS: %0.2f (+/- %0.2f)  Time: %0.5f" % (scores4.mean(), scores4.std(), time4))


# Criando o arquivo .txt com informações do banco e resultados
try:
    arq = open(pathOut + info['name'] + ' - read_me.txt', 'r+')
except FileNotFoundError:
    arq = open(pathOut + info['name'] + ' - read_me.txt', 'w+')
    arq.write('Database: ' + info['name'] + '\n')
    arq.write('Description: ' + info['description'] + '\n')
    arq.write('Autor: ' + info['autor'] + '\n')
    arq.write('Date: ' + info['date'] + '\n\n')
    arq.write('Results{\n')
    arq.write('Distance\tAccuracy\t\tErro\t\t\t\tTime\n' + '_' * 20 + '\n')
    arq.write('Euclidian\t' + str(scores.mean()) + '\t(+/- ' + str(scores.std()) + ')' + '\t'+ str(time1)+'\n')
    arq.write('Manhathan\t' + str(scores2.mean()) + '\t(+/- ' + str(scores2.std()) + ')' + '\t'+ str(time2)+'\n')
    arq.write('Chebyshev\t' + str(scores3.mean()) + '\t(+/- ' + str(scores3.std()) + ')' + '\t'+ str(time3)+'\n')
    arq.write('Mahalanobis\t' + str(scores4.mean()) + '\t(+/- ' + str(scores4.std()) + ')' + '\t'+ str(time4)+'\n')
arq.close()
