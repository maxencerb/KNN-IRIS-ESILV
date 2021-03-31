import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from binary_tree import Node

def readFile(filename, classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
    f = open(filename, 'r')
    data = []
    output = []
    for line in f.readlines():
        line = line[:-1]
        val = line.split(',')
        data.append(list(map(float, val[:-1])))
        output.append(classes.index(val[-1]))
    return np.array(data), np.array(output)

def split_dataset(data, output, test_size = .2):
    # https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=test_size)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=validation_size)
    return x_train, y_train, x_test, y_test#, x_val, y_val

def error(y_true, y_pred):
    return sum((y_true - y_pred) ** 2) ** 1/2


def reduce(x: np.ndarray):
    """
    centre et réduit les données en fonction de leur variance
    """
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    return (x - mean) / var


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


class KNearestNeighbours:

    @staticmethod
    def fit(x_train, y_train, k_neighbours, algo = 'brute', leaf_size = 20):
        return KNearestNeighbours(x_train, y_train, k_neighbours, algo, leaf_size)

    def __init__(self, x_train, y_train, k_neighbours, algo, leaf_size) -> None:
        if algo == 'brute':
            self.x_train = x_train
            self.y_train = y_train
        elif algo == 'kd-tree':
            self.leaf_size = leaf_size
            self.__get_kdtree(x_train, y_train)
        self.k_neighbours = k_neighbours
    
    def __get_kdtree(self, x_train: np.ndarray, y_train: np.ndarray):
        index_var: np.ndarray = np.arange(x_train.shape[1]) # number of columns (features)
        # sort index by variance
        order = np.argsort(x_train.var(0))
        index_var = index_var[order][::-1] # Descending order
        col = 1
        self.tree, childs_x, childs_y = KNearestNeighbours.__split_tree_node(x_train, y_train, 0)
        current_nodes = [self.tree]
        nb_feature = x_train.shape[1]
        while True:
            new_child_x = []
            new_child_y = []
            new_nodes = []
            for i in range(len(current_nodes)):
                node_left, child_x_left, child_y_left = KNearestNeighbours.__split_tree_node(childs_x[2 * i], childs_y[2 * i], col % nb_feature)
                node_right, child_x_right, child_y_right = KNearestNeighbours.__split_tree_node(childs_x[2 * i + 1], childs_y[2 * i + 1], col % nb_feature)
                new_child_x += child_x_left + child_x_right
                new_child_y += child_y_left + child_y_right
                new_nodes += [node_left, node_right]
                current_nodes[i].left = node_left
                current_nodes[i].right = node_right
            childs_x = new_child_x
            childs_y = new_child_y
            current_nodes = new_nodes
            if len(childs_y[0]) / 2 <= self.leaf_size:
                for i in range(len(current_nodes)):
                    current_nodes[i].left = Node(childs_x[2 * i], childs_y[2 * i], isLeaf=True)
                    current_nodes[i].right = Node(childs_x[2 * i + 1], childs_y[2 * i + 1], isLeaf=True)
                break
            col += 1

    @staticmethod
    def __split_tree_node(x: np.ndarray, y: np.ndarray, col: int):
        # sort according to col
        if col >= x.shape[1]: raise ValueError('Must be less than the number of features')
        if x.shape[0] != len(y): raise ValueError('Labels and Features must be the same size')
        order = np.argsort(x[:, col])
        x = x[order]
        y = y[order]
        median = len(x) // 2
        node = Node(x[median], y[median])
        childs_x = [x[:median], x[median + 1:]]
        childs_y = [y[:median], y[median + 1:]]
        return node, childs_x, childs_y

    def get_distance(self, point):
        x_train = self.x_train
        distance = np.ones(x_train.shape) * point
        distance = (distance - x_train) ** 2
        result = [np.sqrt(sum(el)) for el in distance]
        return np.array(result)

    def predict_point(self, point):
        order = np.argsort(self.get_distance(point))
        y_train = self.y_train[order][:self.k_neighbours]
        return sum(y_train) / len(y_train)

    def predict(self, x_test):
        return np.array([self.predict_point(point) for point in x_test])