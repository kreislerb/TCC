import pandas as pd
import numpy as np
from Model.Results import Results
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix


# CRIANDO ARQUIVO DE LOG
log = Results('METRICS/')
log.setTitle("Métricas dos modelos SVM")
log.setDescription(
    "Treinamento e classificação utilizando diversas configurações de SVM no dataset Yale Dataface"
    "o qual foi equalizado e descrito pelo método PCA")

# IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])


def calcule_metrics_svm(kernel, n_c):

    c = np.logspace(-10, 10, n_c, base=10)
    count = 1
    if kernel == 'sigmoid' or kernel == 'rbf':

        gamma = np.logspace(-10, 10, n_c, base=10)

        for i in tqdm(range(n_c)):
            for j in range(n_c):

                model = SVC(C=c[i], gamma=gamma[j], kernel=kernel, random_state=1)
                predict = cross_val_predict(model, database, labels, cv=11)
                metric = classification_report(labels,predict)
                log.insertProcessSpace("Validação Cruzada - Kernel-> {}".format(kernel))
                log.insertProcess("Iteração N°", str(count))
                log.insertProcess("Parâmetros de teste", 'C: {}     Gamma: {}'
                                                        .format(model.C, model.gamma))
                log.insertProcess("Report: ","{}".format(metric))
                count += 1
                del model

    elif kernel == 'linear':


        for i in tqdm(range(n_c)):

            model = SVC(C=c[i], kernel=kernel, random_state=1)
            predict = cross_val_predict(model, database, labels, cv=11)
            metric = classification_report(labels, predict)
            log.insertProcessSpace("Validação Cruzada - Kernel-> {}".format(kernel))
            log.insertProcess("Iteração N°", str(count))
            log.insertProcess("Parâmetros de teste", 'C: {}'
                              .format(model.C))
            log.insertProcess("Report: ","{}".format(metric))
            count += 1

            del model
    else:

        gamma = np.logspace(-10, 10, n_c, base=10)

        for i in tqdm(range(n_c)):
            for j in range(n_c):
                for k in range(2, 6):
                    model = SVC(C=c[i], gamma=gamma[j], degree=k, kernel=kernel, random_state=1)
                    predict = cross_val_predict(model, database, labels, cv=11)
                    metric = classification_report(labels, predict)
                    log.insertProcessSpace("Validação Cruzada - Kernel-> {}".format(kernel))
                    log.insertProcess("Iteração N°", str(count))
                    log.insertProcess("Parâmetros de teste", 'C: {}     Gamma: {}     Degree: {}'
                                      .format(model.C, model.gamma, model.degree))
                    log.insertResultSpace("".format(metric))
                    count += 1
                    del model

    log.save()

calcule_metrics_svm('linear', 100)

