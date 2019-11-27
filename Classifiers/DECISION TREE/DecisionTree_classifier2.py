from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from Model.CreateDataset import CreateDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# IMPORT DATABASE
database = pd.read_csv('../../PCA/PCA_Yale DataBase.csv', index_col='Unnamed: 0')
database.rename(columns={'Unnamed: 0': 'Subjects'}, inplace=True)

dataA = database.iloc[:55]

# RETIRA AS INFORMAÇÕES DE EXPRESSOES E DEIXA APENAS UM NUMERO QUE REPRESENTA UMA PESSOA
data = database.T
labels = list()
for col in data.columns:
    labels.append(col[:2])


def calcule_model_decision_tree(criterion, n, n_samples_split):

    # DATASET
    ds = CreateDataset('CSV/Time_process', "Decision_Tree_{}".format(criterion))


    # DEFININDO PARAMETROS
    #min_inpurity = np.logspace(-20, -1, n, base=10)
    #max_depth = np.arange(2, n)
    #min_samples_leaf = np.arange(5, n_samples_split)
   # min_samples_split = np.arange(5, n_samples_split-2)


    for i in tqdm(np.arange(2, n)):
        for j in (np.arange(1, n_samples_split)):

            ds.set_columns("max_depth min_samples_leaf Accuracy Std Time_process".split())

            t0 = time()
            model = DecisionTreeClassifier(criterion=criterion, random_state=1,
                                           max_depth=i,
                                           min_samples_leaf=j,
                                           min_impurity_decrease=1e-10)

            scores = cross_val_score(model, database, labels, cv=11, scoring='accuracy')

            ds.insert_row([int(model.max_depth),
                           int(model.min_samples_leaf),
                           scores.mean(),
                           scores.std(),
                           time()-t0])
            del model
    ds.save()

# gini || entropy
calcule_model_decision_tree('entropy', 16, 16)

#OBS:
# O que mais influenciu na acurácia, foi a quantidade de classes | a quantidade de amostras por classe
# A quantidade de componentes teve sua influencia



# CART (Árvores de classificação e regressão) é muito semelhante ao C4.5,
# mas difere no fato de suportar variáveis ​​de destino numéricas (regressão)
# e não computar conjuntos de regras. O CART constrói árvores binárias usando
# o recurso e o limite que produzem o maior ganho de informações em cada nó.

# o scikit-learn usa uma versão otimizada do algoritmo CART; no entanto, a
# implementação do scikit-learn não suporta variáveis ​​categóricas por enquanto.

# Criterion "gini" || "entropy"
# min_impurity_decrease : float, optional (default=0.)
# max_leaf_nodes : int or None, optional (default=None)
# max_depth : int or None, optional (default=None)
# min_samples_split : int, float, opcional (padrão = 2)
# max_features : int, float, string ou None, opcional (padrão = None)