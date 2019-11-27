import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlt:

    def __init__(self):
        pass

    def generate_confusion_matrix(y_true, y_pred,
                              normalize=False,
                              title=None,
                              cmap=plt.get_cmap('Blues')):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Matriz de confusão normalizada'
            else:
                title = 'Matriz de confusão'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.rcParams.update({'font.size': 5})
        fig, ax = plt.subplots(figsize=(15, 15), dpi=150)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               #xticklabels=str(classes), yticklabels=(classes),
               title=title+"\n",
               ylabel='Classe verdadeira',
               xlabel='Classe predita')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")


        return ax



















