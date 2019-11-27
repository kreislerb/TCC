from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from Model.CreateDataset import CreateDataset
from Model.Results import Results
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

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


def calcule_model_decision_tree(criterion, n, n_samples_split):

    # DATASET
    ds = CreateDataset('CSV/', "Decision_Tree_{}".format(criterion))

    # DEFININDO PARAMETROS
    min_inpurity = np.logspace(-10, 10, n, base=10)
    max_depth = np.linspace(1, 100, n)
    min_samples_split = np.arange(2, n_samples_split)
    maior_score_encontrado = np.array([0, 0, 0])
    best_params = ''
    count = 1
    time_total= 0

    for i in tqdm(range(n)):
        for j in (range(n)):
            for k in range(n_samples_split-2):
                ds.set_columns("min_impurity max_depth min_samples_split Accuracy Std".split())
                t0 = time()
                log.insertProcessSpace("Validação Cruzada - Criterion-> {}".format(criterion))
                log.insertProcess("Iteração N°", str(count))
                log.insertProcess("Parâmetros de teste", 'min_impurity_decrease: {}     max_depth: {} '
                                                         'min_samples_split: {}'.format(min_inpurity[i], max_depth[j],
                                                                                        min_samples_split[k]))

                model = DecisionTreeClassifier(criterion=criterion, random_state=1,
                                               min_impurity_decrease=min_inpurity[i],
                                               max_depth=max_depth[j],
                                               min_samples_split=min_samples_split[k])
                scores = cross_val_score(model, database, labels, cv=11, scoring='accuracy')

                time_total += time()-t0
                log.insertProcess("Duração do processamento", "%0.3fs" % (time() - t0))

                ds.insert_row([model.min_impurity_decrease, model.max_depth,model.min_samples_split, scores.mean(), scores.std()])

                if scores.mean() > maior_score_encontrado.mean():
                    maior_score_encontrado = scores
                    best_params = "min_impurity_decrease: {}     " \
                                  "max_depth: {}      " \
                                  "min_samples_split: {}".format(min_inpurity[i], max_depth[j], min_samples_split[k])
                log.insertResultAcuracy("Acurácia media / Desvio padão médio", scores)
                count += 1
                del model

    log.insertResultSpace("MELHOR RESULTADO")
    log.insertResult(best_params, maior_score_encontrado)
    log.insertResultTimeMean("Tempo médio por iteração: ", "%0.4fs" % (time_total/count))
    log.insertResultTimeMean("Total de iterações: ", count)
    log.insertResultTimeMean("Tempo total: ", time_total)
    log.save()
    ds.save()

# gini || entropy
calcule_model_decision_tree('gini', 10, 8)

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