import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# IMPORTANDO TABELAS DE RESULTADOS
'''
id_one_1 = pd.read_csv('CSV/NeuralNetwork_identity_one_layers 27-9-2019  4 6 53.csv', index_col='Unnamed: 0')
id_two = pd.read_csv('CSV/NeuralNetwork_identity_two_layers 27-9-2019  9 41 36.csv',index_col='Unnamed: 0')
id_three = pd.read_csv('CSV/NeuralNetwork_identity_three_layers 27-9-2019  9 59 20.csv',index_col='Unnamed: 0')

re_one_1 = pd.read_csv('CSV/NeuralNetwork_relu_one_layers 27-9-2019  4 5 26.csv', index_col='Unnamed: 0')
re_two = pd.read_csv('CSV/NeuralNetwork_relu_two_layers 27-9-2019  9 37 39.csv', index_col='Unnamed: 0')
re_three = pd.read_csv('CSV/NeuralNetwork_relu_three_layers 27-9-2019  14 4 25.csv', index_col='Unnamed: 0')

ta_one_1 = pd.read_csv('CSV/NeuralNetwork_tanh_one_layers 27-9-2019  4 7 28.csv', index_col='Unnamed: 0')
ta_two = pd.read_csv('CSV/NeuralNetwork_tanh_two_layers 27-9-2019  9 30 28.csv', index_col='Unnamed: 0')
ta_three = pd.read_csv('CSV/NeuralNetwork_tanh_three_layers 27-9-2019  12 49 18.csv', index_col='Unnamed: 0')

lo_one_1 = pd.read_csv('CSV/NeuralNetwork_logistic_one_layers 27-9-2019  4 8 3.csv', index_col='Unnamed: 0')
lo_two = pd.read_csv('CSV/NeuralNetwork_logistic_two_layers 27-9-2019  4 15 31.csv', index_col='Unnamed: 0')
lo_three = pd.read_csv('CSV/NeuralNetwork_logistic_three_layers 27-9-2019  11 24 12.csv', index_col='Unnamed: 0')

id_one_1 = pd.read_csv('CSV/Time_process/NeuralNetwork_identity_one_layers 3-11-2019  2 25 25.csv', index_col='Unnamed: 0')
id_two = pd.read_csv('CSV/Time_process/NeuralNetwork_identity_two_layers 3-11-2019  2 32 49.csv',index_col='Unnamed: 0')
id_three = pd.read_csv('CSV/Time_process/NeuralNetwork_identity_three_layers 3-11-2019  4 31 46.csv',index_col='Unnamed: 0')

re_one_1 = pd.read_csv('CSV/Time_process/NeuralNetwork_relu_one_layers 3-11-2019  2 26 25.csv', index_col='Unnamed: 0')
re_two = pd.read_csv('CSV/Time_process/NeuralNetwork_relu_two_layers 3-11-2019  3 41 24.csv', index_col='Unnamed: 0')
re_three = pd.read_csv('CSV/Time_process/NeuralNetwork_relu_three_layers 3-11-2019  7 56 0.csv', index_col='Unnamed: 0')

ta_one_1 = pd.read_csv('CSV/Time_process/NeuralNetwork_tanh_one_layers 3-11-2019  2 26 2.csv', index_col='Unnamed: 0')
ta_two = pd.read_csv('CSV/Time_process/NeuralNetwork_tanh_two_layers 3-11-2019  3 26 16.csv', index_col='Unnamed: 0')
ta_three = pd.read_csv('CSV/Time_process/NeuralNetwork_tanh_three_layers 3-11-2019  4 55 53.csv', index_col='Unnamed: 0')

lo_one_1 = pd.read_csv('CSV/Time_process/NeuralNetwork_logistic_one_layers 3-11-2019  2 25 46.csv', index_col='Unnamed: 0')
lo_two = pd.read_csv('CSV/Time_process/NeuralNetwork_logistic_two_layers 3-11-2019  3 13 16.csv', index_col='Unnamed: 0')
lo_three = pd.read_csv('CSV/Time_process/NeuralNetwork_logistic_three_layers 3-11-2019  4 44 59.csv', index_col='Unnamed: 0')
'''

id_one_1 = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_identity_one_layers 3-11-2019  20 5 55.csv', index_col='Unnamed: 0')
id_two = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_identity_two_layers 3-11-2019  20 3 37.csv',index_col='Unnamed: 0')
id_three = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_identity_three_layers 3-11-2019  19 48 49.csv',index_col='Unnamed: 0')

re_one_1 = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_relu_one_layers 3-11-2019  20 4 10.csv', index_col='Unnamed: 0')
re_two = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_relu_two_layers 3-11-2019  19 51 56.csv', index_col='Unnamed: 0')
re_three = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_relu_three_layers 3-11-2019  19 13 46.csv', index_col='Unnamed: 0')

ta_one_1 = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_tanh_one_layers 4-11-2019  18 22 18.csv', index_col='Unnamed: 0')
ta_two = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_tanh_two_layers 3-11-2019  20 0 51.csv', index_col='Unnamed: 0')
ta_three = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_tanh_three_layers 3-11-2019  19 25 47.csv', index_col='Unnamed: 0')

lo_one_1 = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_logistic_one_layers 3-11-2019  20 5 30.csv', index_col='Unnamed: 0')
lo_one_1_ORL = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_logistic_one_layers 17-11-2019  22 20 48.csv', index_col='Unnamed: 0')

lo_two = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_logistic_two_layers 3-11-2019  20 3 2.csv', index_col='Unnamed: 0')
lo_three = pd.read_csv('CSV/Time_process/Alpha_NeuralNetwork_logistic_three_layers 3-11-2019  19 43 20.csv', index_col='Unnamed: 0')




def show_graph_result(activation, n_layers, results):
    # ONE_HIDDEN_LAYER

    if n_layers == 1:
        # PLOT GRÁFICO

        x = np.array(results[0]['hidden_layer'], dtype=int)
        accuracy_mean_list = list()
        for i in range(len(results)):
            y = np.array(results[i]['Accuracy'], dtype=float)

            # identificando pontos de máximo
            max_value = np.max(y)
            max_index = results[i]['Accuracy'].idxmax()
            hidden_max = x[max_index]
            accuracy_mean = y.mean()
            accuracy_mean_list.append(accuracy_mean)


            plt.plot(x, y, alpha=.7, label=activation[i] +
                                 ' ' * (17 - len(activation[i])) +
                                 "(Máx = {:2.3f})      Média: {:2.3f}".format(max_value, accuracy_mean))

        plt.ylim(0.75, 1.017)
        plt.margins(x=0)
        plt.ylabel("Acurácia")
        plt.xlabel("Quantidade de neurônios na camada oculta (x)")
        plt.title("Acurácia em função do numero de neurônios da HL em uma RNA com uma HL")
        plt.legend(loc=4)

    # TWO LAYERS
    elif n_layers == 2:

        i = results['hidden_layer'].apply(lambda x: "{}".format(x.split(',')[0][1:]))
        j = results['hidden_layer'].apply(lambda x: "{}".format(x.split(',')[1][:-1]))
        i = np.array(i.unique(), dtype=int)
        j = np.array(j.unique(), dtype=int)

        accuracy = np.array(results["Accuracy"], dtype=float)
        accuracy = np.reshape(accuracy, (len(i), len(j)))

        # AJUSTANDO O GRÁFICO
        df = pd.DataFrame(accuracy, i, j)

        fig = plt.figure(figsize=(25, 25))

        ax = sns.heatmap(df, cmap="Greys", annot=True, vmin=.9, vmax=1,
                         figure=fig, center=.95, fmt=".3f", square=False,
                         cbar_kws={'label': 'Acurácia'})

        ax.figure.subplots_adjust(bottom=0.15)
        ax.set_title("Acurácia dos modelos de RNA's com duas HL's a partir da variação da \n quantidade de neurônios de cada HL utilizando uma AF {}\n".format(activation))
        ax.invert_yaxis()
        ax.tick_params(axis='x', pad=0, labelsize=10, labelrotation=45)
        ax.tick_params(axis='y', pad=0, labelsize=10, labelrotation=0)

        ax.set_xlabel("Quantidade de neurônios na HL 1 (x)")
        ax.set_ylabel("Quantidade de neurônios na HL 2 (y)")
        ax.set_xticks(np.arange(i.shape[0]))
        ax.set_yticks(np.arange(j.shape[0]))
        ax.set_xticklabels(i)
        ax.set_yticklabels(j)

    # ax.set_ylim(-10, 10)
    # ax.set_xlim(-10, 10)

    # THREE LAYERS
    else:

        #Diminuindo a quantidade de amostras

       # condition = results.index % 2 == 0
       #esults = results[condition]

        print(results.head())
        print(results.describe())

        # SEPARANDO AS VARIAVEIS E CONVERTENDO PARA SEU DEFIDO TIPO
        x = results['hidden_layer'].apply(lambda x: "{}".format(x.split(',')[0][1:]))
        y = results['hidden_layer'].apply(lambda x: "{}".format(x.split(',')[1]))
        z = results['hidden_layer'].apply(lambda x: "{}".format(x.split(',')[2][:-1]))

        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        z = np.array(z, dtype=int)

        c = np.array(results["Accuracy"], dtype=float)

        # CRIANDO GRÁFICO 3D
        fig = plt.figure()
        ax = Axes3D(fig)

        # Ajusta os labels dos marcadores
        ax.tick_params(axis='x', pad=-5, labelsize=10, labelrotation=45)
        ax.tick_params(axis='y', pad=-5, labelsize=10, labelrotation=45)
        ax.set_xlabel("Quantidade de Nerônios na Camada 1 (x)")
        ax.set_ylabel("Quantidade de Nerônios na Camada 2 (y)")
        ax.set_zlabel("Quantidade de Nerônios na Camada 3 (z)")
        ax.view_init(12, -120)

        graf = ax.scatter(x, y, z, c=c, cmap="gnuplot_r", s=40, alpha=1, vmax=1)
        fig.colorbar(graf, shrink=.8, aspect=20, anchor=(-.8, 0.5), label="Acurácia")

        ax.set_title("Acurácia dos modelos de RNA's com três HL's a partir da variação da \n quantidade de neurônios de cada HL utilizando uma AF {}".format(activation))


    plt.show()




# THREE LAYERS GRAPH
#show_graph_result('Logistica', 3, lo_three)
#show_graph_result('ReLU', 3, re_three)
#show_graph_result('Tanh', 3, ta_three)
#show_graph_result('Identidade', 3, id_three)

# TWO LAYERS
#show_graph_result('Logistica', 2, lo_two)
#show_graph_result('ReLu', 2, re_two)
#show_graph_result('Tanh', 2, ta_two)
#show_graph_result('Identidade', 2, id_two)


# ONE LAYER
'''
show_graph_result(['identity', 'relu'+' '*15, 'logistic'+' '*10, 'tanh'+' '*14],
                  1,
                  [id_one_1, re_one_1, lo_one_1, ta_one_1])'''
show_graph_result(['Logistic_YALE', 'Logistic_ORL'],
                  1,
                  [lo_one_1, lo_one_1_ORL ])



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

    print("RNA - 1 HL - LOGISTIC")
    report(lo_one_1)
    print("RNA - 2 HL - LOGISTIC")
    report(lo_two)
    print("RNA - 3 HL - LOGISTIC")
    report(lo_three)
    print("RNA - 1 HL - IDENTIDADE")
    report(id_one_1)
    print("RNA - 2 HL - IDENTIDADE")
    report(id_two)
    print("RNA - 3 HL - IDENTIDADE")
    report(id_three)
    print("RNA - 1 HL - RELU")
    report(re_one_1)
    print("RNA - 2 HL - RELU")
    report(re_two)
    print("RNA - 3 HL - RELU")
    report(re_three)
    print("RNA - 1 HL - TANH")
    report(ta_one_1)
    print("RNA - 2 HL - TANH")
    report(ta_two)
    print("RNA - 3 HL - TANH")
    report(ta_three)

info()