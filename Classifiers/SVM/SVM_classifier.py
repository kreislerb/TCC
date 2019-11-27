import pandas as pd
from time import time
import numpy as np
from Model.Results import Results
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# CABEÇALHO

log = Results('RESULTS/')
log.setTitle("Classificadores SVM")
log.setDescription("Treinamento e classificação utilizando diversas configurações de um SVM, no dataset Yale Dataface"
                   "o qual foi equalizado e descrito pelo método PCA")

#IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])

c = np.logspace(-10, 10, 10, base=10)
gamma = np.logspace(0, -10, 10, base=10)
maiorScoreEncontrado = np.array([0, 0, 0])
bestParams = ''
count = 0
for i in tqdm(range(10)):
    for j in range(10):
            t0 = time()
            log.insertProcessSpace("Validação Cruzada - KERNEL-> Polinomial")
            log.insertProcess("Iteração N°", str(count))
            log.insertProcess("Parâmetros de teste", "C: {}  Gamma: {} ".format(c[i], gamma[j]))
            model = SVC(C=c[i], kernel='rbf', gamma=gamma[j],random_state=45,coef0)
            scores = cross_val_score(model, database, labels, cv=11, scoring='accuracy')
            if scores.mean() > maiorScoreEncontrado.mean():
                maiorScoreEncontrado = scores
                bestParams = "C: {}     Gamma: {}".format(c[i], gamma[j])

            log.insertProcess("Duração do processamento", "%0.3fs" % (time() - t0))
            log.insertResultAcuracy("Acurácia media / Desvio padão médio", scores)
            count += 1
            del model
log.insertResultSpace("MELHOR RESULTADO")
log.insertResult(bestParams, maiorScoreEncontrado)
log.save()

'''
c = np.logspace(-6, 6, 20, base=10)
gamma = np.logspace(0, -6, 20, base=10)
maiorScoreEncontrado = np.array([0, 0, 0])
bestParams = ''
count = 0
for i in range(20):
    for j in range(20):
        for k in range(2,6):
            t0 = time()
            log.insertProcessSpace("Validação Cruzada - KERNEL-> Polinomial")
            log.insertProcess("Iteração N°", str(count))
            log.insertProcess("Parâmetros de teste", "C: {}  Gamma: {}  Degree: {}".format(c[i], gamma[j], k))
            model = SVC(C=c[i], kernel='poly', gamma=gamma[j], degree = k,random_state=45)
            scores = cross_val_score(model, database, labels, cv=11, verbose=2, scoring='accuracy')
            if scores.mean() > maiorScoreEncontrado.mean():
                maiorScoreEncontrado = scores
                bestParams = "C: {}     Gamma: {}   Degree: {}".format(c[i], gamma[j], k)

            log.insertProcess("Duração do processamento", "%0.3fs" % (time() - t0))
            log.insertResultAcuracy("Acurácia media / Desvio padão médio", scores)
            count += 1
            del model
log.insertResultSpace("MELHOR RESULTADO")
log.insertResult(bestParams, maiorScoreEncontrado)
log.save()
'''


'''
maiorScoreEncontrado = np.array([0, 0, 0])
bestParams = ''
count = 0
c = 4.281332398719396e-06
gamma = 0.006158482110660267
for i in range(2,7):
    t0 = time()
    log.insertProcessSpace("Validação Cruzada - KERNEL-> Polinomial")
    log.insertProcess("Iteração N°", str(count))
    log.insertProcess("Parâmetros de teste", "C: {}  Gamma: {}  Degree: {}".format(c, gamma, i))
    model = SVC(C=c, kernel='poly', gamma=gamma, random_state=45,degree=i)
    scores = cross_val_score(model, database, labels, cv=11, verbose=2, scoring='accuracy')
    if scores.mean() > maiorScoreEncontrado.mean():
        maiorScoreEncontrado = scores
        bestParams = 'C = ' + str(c) + '     Gamma = ' + str(gamma) + "     Degree = "+ str(i)

    log.insertProcess("Duração do processamento", "%0.3fs" % (time() - t0))
    log.insertResultAcuracy("Acurácia media / Desvio padão médio", scores)
    count += 1
    del model
log.insertResultSpace("MELHOR RESULTADO")
log.insertResult(bestParams, maiorScoreEncontrado)
log.save()
'''


'''
params_grid = {
        'kernel': ('linear', 'sigmoid'),
        'C': c,
        'gamma': gamma
}
grid = GridSearchCV(SVC(), params_grid, cv=11, refit=True)
grid.fit(database, labels)
log.insertProcess("Melhores parâmetros encontrados", str(grid.best_params_))
log.insertProcess("Melhor Score", str(grid.best_score_))
log.insertProcess("Duração do processamento", "%0.3fs" % (time() - t0))
log.save()
'''













