import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
CLASSES_2 = ['classA']
FEATURES = ['sepal length (in cm)', 'sepal width (in cm)', 'petal length (in cm)', 'petal width (in cm)']

def readFile(filename, classes = None, skip_output = False) -> tuple:
    """
    Lit le fichier contenant les données IRIS

    Arguments
    ---------
    filename:     chemin vers le fichier

    classes:      Liste contenant les différents labels

    skip_output:  Ne prend pas en compte la dernière colonne comme output si True

    Return
    ------

    data:     2D numpy array avec les features

    output:   1D numpy array contenant les labels
    """
    f = open(filename, 'r')
    data = []
    output = []
    for line in f.readlines():
        line = line[:-1]
        val = line.split(',')
        if skip_output:
            data.append(list(map(float, val)))
        else:
            data.append(list(map(float, val[:-1])))
            output.append(classes.index(val[-1]))
    return np.array(data), np.array(output)

def split_dataset(data, output, test_size = .2) -> tuple:
    """
    Sépare le dataset aléatoirement en données d'entraînement et de test

    Argument
    --------
    data:       2D numpy array avec les features

    output:     1D numpy array contenant les labels

    test_size:  Taille en pourcentage des données de test

    Return
    ------
    x_train: features entrainement
    
    y_train: labels entrainement
    
    x_test:  features test
    
    y_test:  labels test

    Citation
    --------
    https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros

    """
    x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=test_size)
    return x_train, y_train, x_test, y_test

def error(y_true, y_pred):
    return sum(y_true != y_pred)


def reduce(x: np.ndarray) -> np.ndarray:
    """
    Centre et réduit les données en fonction de leur variance par colonne

    Arguments
    ---------
    x: 2D numpy array

    Usage
    -----
    ```python
    x = np.random.random(9).reshape((3, 3))
    x = reduce(x)
    ```
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    ---------
    ```python
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    ```

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def saveResults(filename, results, CLASSES):
    if min(results) < 0 or max(results) >= len(CLASSES):
        raise ValueError('values must be index between 0 and the number of classes (excluded)')
    results = list(map(lambda x: CLASSES[x], results))
    line = '\n'.join(results)
    f = open(filename, 'w')
    f.write(line)
    f.close()