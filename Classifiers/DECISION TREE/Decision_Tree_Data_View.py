import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



# IMPORTANDO TABELAS DE RESULTADOS
entropy = pd.read_csv('CSV/Decision_Tree_entropy 12-10-2019  16 14 5.csv', index_col='Unnamed: 0')
gini = pd.read_csv('CSV/Decision_Tree_gini 12-10-2019  16 40 51.csv', index_col='Unnamed: 0')

gini_leaf = pd.read_csv('CSV/Time_processDecision_Tree_gini 14-11-2019  0 0 45.csv', index_col='Unnamed: 0')
gini_split = pd.read_csv('CSV/Time_processDecision_Tree_gini 13-11-2019  20 1 48.csv', index_col='Unnamed: 0')
gini_split_leaf = pd.read_csv('CSV/Time_processDecision_Tree_gini 13-11-2019  23 43 48.csv', index_col='Unnamed: 0')


entropy_leaf = pd.read_csv('CSV/Time_processDecision_Tree_entropy 14-11-2019  0 3 57.csv', index_col='Unnamed: 0')
entropy_split = pd.read_csv('CSV/Time_processDecision_Tree_entropy 13-11-2019  20 5 1.csv', index_col='Unnamed: 0')
entropy_split_leaf = pd.read_csv('CSV/Time_processDecision_Tree_entropy 13-11-2019  23 42 44.csv', index_col='Unnamed: 0')

def show_graph_result(criterio, results):
    #print(results.describe())
   # print(results)

    #results['min_impurity'] = results['min_impurity'].apply(lambda x: "{:2.2f}".format(np.log10(x)))

    # Convertendo dados para float, se não o gráfico ficará com escala desajustada
    x = results['max_depth']
    y = results['min_samples_leaf']
    x = np.array(x.unique(), dtype=int)
    y = np.array(y.unique(), dtype=int)


    accuracy = np.array(results["Accuracy"])
    accuracy = np.reshape(accuracy, (len(x), len(y)))

    # AJUSTANDO O GRÁFICO
    df = pd.DataFrame(accuracy, x, y)

    fig = plt.figure(figsize=(20, 15))

    ax = sns.heatmap(df, cmap="Greys", annot=True, vmax=.7,
                     figure=fig, fmt=".3f", square=False,
                     cbar_kws={'label': 'Acurácia'})

    ax.figure.subplots_adjust(bottom=0.15)
    ax.set_title("Acurácia dos modelos obtidos com a variação dos parâmetros utilizando Criterio {}".format(criterio))
    ax.invert_yaxis()
    ax.tick_params(axis='x', pad=10, labelsize=10, labelrotation=45)
    ax.tick_params(axis='y', pad=10, labelsize=10, labelrotation=0)

    #ax.set_xlabel("Min_Amostras_divisao")
    ax.set_xlabel("Min_Amostras_Folha")
    ax.set_ylabel("Max_Profundidade")
    #ax.set_ylabel("Min_Amostras_Divisao")

    plt.show()
'''
    x = np.array(results['max_depth'], dtype=float)
    y = np.array(results['min_samples_leaf'], dtype=float)
    #z = np.array(results['min_impurity'], dtype=float)
    c = np.array(results['Accuracy'], dtype=float)

    fig = plt.figure()
    ax = Axes3D(fig)

    # Ajusta os labels dos marcadores

    ax.tick_params(axis='x', pad=-5, labelsize=10, labelrotation=45)
    ax.tick_params(axis='y', pad=-5, labelsize=10, labelrotation=45)

    ax.set_xlabel("min_samples_leaf",  labelpad=10)
    ax.set_ylabel("max_depth", labelpad=10)
    ax.set_zlabel("log(min_impurity)", labelpad=5)
    ax.view_init(12, -120)

    graf = ax.scatter(x, y, z, c=c, cmap="gnuplot_r", s=40, alpha=1)

    cb = fig.colorbar(graf, shrink=.8, aspect=20, anchor=(-.8, 0.5))
    cb.set_label("Acurácia", labelpad = 15)

    #fig.colorbar(graf, shrink=0.8, aspect=10 ,anchor=(-.8, 0.5))

    ax.set_title('Acurácia dos modelos obtidos com a variação dos parâmetros utilizando Criterio {}'.format(criterio))
    '''



# PLOT COM PCA_60
#show_graph_result('gini_pca_60', gini_pca_60)
#show_graph_result('gini_5_c', gini_5_c)

# PLOT COM PCA_70
#show_graph_result('gini_pca_70', gini_pca_70)
#show_graph_result('entropy_pca_70', entropy_pca_70)

# PLOT COM PCA_90
show_graph_result('Entropia', entropy_leaf)
#show_graph_result('Gini', gini_leaf)

#show_graph_result('entropy', entropy)

# PLOT COM PCA_95
#show_graph_result('gini_PCA_70', gini_pca_95)
#show_graph_result('entropy_PCA_70', entropy_pca_70)





#   RELATORIO FINAL

def calc_time_mean(database):
    print(":: Análise Geral dos modelos ::")
    print("\t * Tempo médio de treino: {:.3}s   +/- {:.2} s".format(database["Time_process"].mean(),
                                               database["Time_process"].std()))
def calc_accuracy_mean_total(database):
    print("\t * Acurácia média: {:.4}  Desvio: +/- {:.3}".format(database['Accuracy'].mean(),
                                                           database['Accuracy'].std()))

def calc_acuracy_max_and_min_std_and_min_time(database):
    print(":: Escolha do melhor modelo ::")
    filter_accuracy = database['Accuracy'] >= database['Accuracy'].max()
    acuracias_filtradas = database[filter_accuracy]
    filter_std = acuracias_filtradas['Std'] <= acuracias_filtradas['Std'].min()
    desvios_filtrados = acuracias_filtradas[filter_std]
    filter_time = desvios_filtrados['Time_process'] <= desvios_filtrados['Time_process'].min()
    tempos_filtrados = desvios_filtrados[filter_time]
    print(tempos_filtrados.head())
  #  print(new.head())
    linha = '-_-' * 25
    print(linha)

def report(database):
    linha = '-_-' * 25
    print(linha)
    calc_time_mean(database)
    calc_accuracy_mean_total(database)
    calc_acuracy_max_and_min_std_and_min_time(database)


# Analise dos dados
def info():
    print(entropy_leaf.describe())
    print("DT_ENTROPY")
    report(entropy_leaf)
    print("DT_GINI")
    report(gini_leaf)


info()