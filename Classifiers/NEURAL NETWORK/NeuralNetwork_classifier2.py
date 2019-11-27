import pandas as pd
from tqdm import tqdm
from time import time
import numpy as np
from Model.CreateDataset import CreateDataset
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score



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

data2 = database2.T
labels2 = list()
for col in data2.columns:
    labels2.append(col)


def calcule_model_ann(activation, n_layers, database, labels):

    # CABEÇALHO
    ds = CreateDataset('CSV/Time_process/', "Alpha_NeuralNetwork_{}_{}_layers".format(activation, n_layers))


    for i in tqdm(np.arange(2, 121, 2)):
        #for j in np.arange(20, 91, 10):
            #for k in np.arange(20, 91, 20):

            ds.set_columns("hidden_layer Accuracy Std Time_train".split())
            model = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(i), random_state=1, activation=activation)
            t0 = time()
            scores = cross_val_score(model, database2, labels2, cv=10, scoring='accuracy')
            ds.insert_row([model.hidden_layer_sizes, scores.mean(), scores.std(), time()-t0])
            del model

    ds.save()
    del ds


#calcule_model_ann('identity', 'one')
calcule_model_ann('logistic', 'one', database, labels)
#calcule_model_ann('tanh', 'one')
#calcule_model_ann('relu', 'one')

#calcule_model_ann('identity', 'two')
#calcule_model_ann('logistic', 'two')
#calcule_model_ann('tanh', 'two')
#calcule_model_ann('relu', 'two')

#calcule_model_ann('identity', 'three')
#calcule_model_ann('logistic', 'three')
#calcule_model_ann('tanh', 'three')
#calcule_model_ann('relu', 'three')



'''
activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
Activation function for the hidden layer.

‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
‘relu’, the rectified linear unit function, returns f(x) = max(0, x)


solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
The solver for weight optimization.

‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
‘sgd’ refers to stochastic gradient descent.
‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba


learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
Learning rate schedule for weight updates.

‘constant’ is a constant learning rate given by ‘learning_rate_init’.
‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
Only used when solver='sgd'.

learning_rate_init : double, optional, default 0.001
The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.

power_t : double, optional, default 0.5
The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.

max_iter : int, optional, default 200
Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.

shuffle : bool, optional, default True
Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.

random_state : int, RandomState instance or None, optional, default None
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

tol : float, optional, default 1e-4
Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.

verbose : bool, optional, default False
Whether to print progress messages to stdout.

warm_start : bool, optional, default False
When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. See the Glossary.

momentum : float, default 0.9
Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.

nesterovs_momentum : boolean, default True
Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.

early_stopping : bool, default False
Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’

validation_fraction : float, optional, default 0.1
The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True

beta_1 : float, optional, default 0.9
Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’

beta_2 : float, optional, default 0.999
Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’

epsilon : float, optional, default 1e-8
Value for numerical stability in adam. Only used when solver=’adam’

n_iter_no_change : int, optional, default 10
Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’
'''