from time import time
import pandas as pd
import matplotlib.pyplot as plt

from Model.Results import Results
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.model_selection import train_test_split

# CABEÇALHO

log = Results('RESULTS/')
log.setTitle("Classificadores de Árvore de Decisão")
log.setDescription(
    "Treinamento e classificação utilizando diversas configurações de árvore de decisao, no dataset Yale Dataface"
    "o qual foi equalizado e descrito pelo método PCA")

# IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

dataA = database.iloc[:55]

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])



X_train, X_test, y_train, y_test = train_test_split(database, labels, test_size=0.1)
model = DecisionTreeClassifier(criterion='gini', random_state=1,min_samples_leaf=5, min_samples_split=5, max_depth=10)
model = model.fit(X_train, y_train)

tree.plot_tree(model, filled=True)
plt.show()






