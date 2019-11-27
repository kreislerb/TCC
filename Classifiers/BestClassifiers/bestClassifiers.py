import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sklearn.neighbors as ng
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from time import time

from Utils.confusion_matrix_cv import ConfusionMatrixPlt as c_matrix

#IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

database2 = pd.read_csv('../../PCA/CSV/PCA_ATT_DATABASE_90.csv', index_col='Unnamed: 0')
database2.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA

data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])

def process_labels(database):
    data = database.T
    labels = list()
    for col in data.columns:
        labels.append(col)
    return labels



labels2 = process_labels(database2)
cv = 10
linha = "-"*40+"\n"

def plt_best_results(title, model, database, labels):


    predict = cross_val_predict(model, database, labels, cv=cv)
    results = cross_validate(model, database, labels, cv=cv, return_train_score=True)
    t0 = time()
    score = cross_val_score(model, database, labels, cv=cv)
    tempo_process = time()-t0


    print(linha+"*\t"+title+" - CLASSIFICATION REPORT\n"+linha)

    score_train =  results['train_score']
    score_test = results['test_score']
    print("Train Score: {} %     +/-  {}".format(score_train*100, score_train.std()))
    print("Train Score: {} %     +/-  {}".format(score_train.mean()*100, score_train.std()))
    print("Test Score: {} %     +/-  {}".format(score_test*100, score_test.std()))
    print("Test Score: {} %     +/-  {}".format(score_test.mean()*100, score_test.std()))
    print("Tempo de processo com CrossValidation: {} s".format(tempo_process))
    print("Tempo de Treinamento: {} s".format(results['fit_time'].mean()))
    print("Tempo de Teste: {} s".format(results['score_time'].mean()))



    print(classification_report(labels,predict))
    #c_matrix.generate_confusion_matrix(np.asarray(labels), predict, title="Matriz de Confusão - "+title)

# BEST METRICS

model_metrics = ng.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
plt_best_results("Metrica Euclidiana", model_metrics, database2, labels2)

# BEST DT

max_depth = 35
min_leaf = 3
model_dt = DecisionTreeClassifier(criterion='gini', random_state=1,
                                           max_depth=max_depth,
                                           min_samples_leaf=min_leaf,
                                           min_impurity_decrease=1e-10)
plt_best_results("Árvore de decisão", model_dt, database2, labels2)

# BEST SVM
c = 4.328761e-17
model_svm = SVC(C=c, kernel='linear', random_state=1)
#plt_best_results("SVM", model_svm,database2, labels2)

# BEST RNA
hl = (66)
activation = 'logistic'
model_rna = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=hl, random_state=1, activation=activation)
#plt_best_results("RNA", model_rna, database2, labels2)


plt.show()





# Plot normalized confusion matrix
#plot_confusion_matrix(labels, predict, normalize=True, title='Normalized confusion matrix')
