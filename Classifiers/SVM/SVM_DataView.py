import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# IMPORTANDO TABELAS DE RESULTADOS
linear = pd.read_csv('CSV/SVM_linear 9-10-2019  11 28 23.csv', index_col='Unnamed: 0')
linear2 = pd.read_csv('CSV/Time_processSVM_linear 17-11-2019  22 5 11.csv', index_col='Unnamed: 0')
sigmoid = pd.read_csv('CSV/SVM_sigmoid 9-10-2019  12 23 19.csv', index_col='Unnamed: 0')
rbf = pd.read_csv('CSV/SVM_rbf 9-10-2019  12 23 55.csv', index_col='Unnamed: 0')
poly = pd.read_csv('CSV/SVM_poly 9-10-2019  12 27 48.csv', index_col='Unnamed: 0')


def show_graph_result(kernel, results):
    print(results.describe())
    print(results.head())
    # KERNEL LINEAR
    if kernel == 'linear':
        # PLOT GRÁFICO
        plt.axes(xscale='log')

        x = np.array(results['C'], dtype=float)
        y = np.array(results['Accuracy'], dtype=float)
        y2 = np.array(linear2['Accuracy'], dtype=float)

        max_value = np.max(y)
        max_index = results['Accuracy'].idxmax()
        c_max = x[max_index]
        accuracy_mean = y.mean()


        plt.plot(x, y, label="SVM Linear_YALE"+ "    (Máx = {:2.3f},   C =  {:2.2} )      Média: {:2.3f}".format(max_value,c_max , accuracy_mean))
       # plt.plot(x, y2, label="SVM Linear_ORL" + "    (Máx = {:2.3f},   C =  {:2.2} )      Média: {:2.3f}".format(max_value,c_max , accuracy_mean))
        plt.plot(x, np.ones(len(x)), label="Acurácia Máxima")
        plt.margins(x=0)
        plt.ylabel("Acurácia")
        plt.xlabel("C")
        plt.title("Acurácia dos modelos obtidos com a variação do parâmetro C do Kernel Linear")
        plt.legend(loc=4)

    # KERNEL SIGMOID || RBF
    elif kernel == 'sigmoid' or kernel == 'rbf':
        # PASSANDO OS PARÃMETROS GAMMA E C PARA ESCALA LOGARÍTIMICA
        results['C'] = results['C'].apply(lambda x: "{:3.2f}".format(np.log10(x)))
        results['Gamma'] = results['Gamma'].apply(lambda x: "{:3.2f}".format(np.log10(x)))

        # COLUNAS UTILIZADAS
        results = results[["C", "Gamma", "Accuracy"]]
        # ADAPTANDO DADOS PARA O GRÁFICO

        col = np.array(results["Gamma"].unique(), dtype=float)
        index = np.array(results["C"].unique(), dtype=float)
        accuracy = np.array(results["Accuracy"], dtype=float)
        accuracy = np.reshape(accuracy, (10, 10))


        # AJUSTANDO O GRÁFICO
        df = pd.DataFrame(accuracy, index, col, dtype=float)

        if kernel == 'rbf':
            title = 'RBF'
        else:
            title = 'Sigmoidal'

        fig = plt.figure(figsize=(16, 9))

        ax = sns.heatmap(df, cmap="Greys", annot=True,  vmin=.55, vmax=1, figure=fig,
                         center=.85, fmt=".3f", square=False,
                         cbar_kws={'label': 'Acurácia'})


        ax.figure.subplots_adjust(bottom=0.15)

        ax.set_title("Acurácia dos modelos obtidos com a variação dos parâmetros com Kernel {}\n".format(title))
        ax.invert_yaxis()
        ax.tick_params(axis='x',  pad=0, labelsize=10, labelrotation=45)
        ax.tick_params(axis='y', pad=0, labelsize=10, labelrotation=0)
        ax.set_ylabel("Log(C)")
        ax.set_xlabel("Log(Gamma)")
        ax.set_xticks(np.arange(col.shape[0]))
        ax.set_yticks(np.arange(index.shape[0]))
        ax.set_xticklabels(col)
        ax.set_yticklabels(index)


        
       # ax.set_ylim(-10, 10)
       # ax.set_xlim(-10, 10)


    # KERNEL POLINOMIAL
    else:

        results['C'] = results['C'].apply(lambda x: "{:2.2f}".format(np.log10(x)))
        results['Gamma'] = results['Gamma'].apply(lambda x: "{:2.1f}".format(np.log10(x)))

        #Convertendo dados para float, se não o gráfico ficará com escala desajustada
        x = np.array(results['Gamma'], dtype=float)
        y = np.array(results['C'], dtype=float)
        z = np.array(results['Degree'], dtype=float)
        c = np.array(results['Accuracy'], dtype=float)

        fig = plt.figure()
        ax = Axes3D(fig)

        #Ajusta os labels dos marcadores
        ax.tick_params(axis='x',  pad=-5, labelsize=10, labelrotation=45)
        ax.tick_params(axis='y', pad=-5, labelsize=10, labelrotation=45)
        ax.set_xlabel("Log(Gamma)")
        ax.set_ylabel("Log(C)")
        ax.set_zlabel("Degree")
        ax.view_init(12, -120)
        graf = ax.scatter(x, y, z, c=c, cmap="gnuplot_r", s=40, alpha=1)

        cb = fig.colorbar(graf, shrink=.8, aspect=20, anchor=(-.8, 0.5))
        cb.set_label("Acurácia", labelpad=15)


        ax.set_title('Acurácia dos modelos obtidos com a variação dos parâmetros do Kernel Polinomial')

    plt.show()

# ESCOLHE GRAFICO PARA PLOTAGEM
show_graph_result('linear', linear)











def calc_time_mean(database):
    print(":: Análise Geral dos modelos ::")
    print("\t * Tempo médio de treino: {:.3}s   +/- {:.2} s".format(database["Time_train"].mean(),
                                               database["Time_train"].std()))
def calc_accuracy_mean_total(database):
    print("\t * Acurácia média: {:.4}  Desvio: +/- {:.3}".format(database['Accuracy'].mean(),
                                                           database['Accuracy'].std()))

def calc_acuracy_max_and_min_std_and_min_time(database):
    print(":: Escolha do melhor modelo ::")
    filter_accuracy = database['Accuracy'] >= database['Accuracy'].max()
    acuracias_filtradas = database[filter_accuracy]
    filter_std = acuracias_filtradas['Std'] <= acuracias_filtradas['Std'].min()
    desvios_filtrados = acuracias_filtradas[filter_std]
    filter_time = desvios_filtrados['Time_train'] <= desvios_filtrados['Time_train'].min()
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
    linear = pd.read_csv('CSV/Time_processSVM_linear 2-11-2019  17 2 39.csv', index_col='Unnamed: 0')
    sigmoid = pd.read_csv('CSV/Time_processSVM_sigmoid 31-10-2019  19 18 41.csv', index_col='Unnamed: 0')
    rbf = pd.read_csv('CSV/Time_processSVM_rbf 31-10-2019  19 17 47.csv', index_col='Unnamed: 0')
    poly = pd.read_csv('CSV/Time_processSVM_poly 31-10-2019  19 41 31.csv', index_col='Unnamed: 0')


    print("SVM - LINEAR")
    report(linear)
    print("SVM - SIGMOIDAL")
    report(sigmoid)
    print("SVM - RBF")
    report(rbf)
    print("SVM - POLY")
    report(poly)


info()





