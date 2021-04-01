from enum import Enum
import numpy as np
from binary_tree import Node

class Algo(Enum):
    """
    Different type of available algoritms for k-nn method

    Values
    ------
    BRUTE   : brute algorithm (array data structure)

    KD_TREE : kd tree algorithm (binary tree data structure)

    Usage
    -----
    ```python
    algo = Algo.KD_TREE
    knn = KNearestNeighbours.fit(x_train, y_train, k_neighbours = 5, algo = algo, leaf_size = 20)
    ...
    ```
    """
    BRUTE = 'brute'
    KD_TREE = 'kd-tree'

class KNearestNeighbours:
    """
    Define the k-nn algorithm

    Methods
    -------
    fit     : fit the algorithm with the training values and label

    predict : predict the labels of a given test sample

    Usage
    -----
    ```python
    from k_nearest_neighbours import Algo

    x_train, y_train, x_test, y_test = ...

    knn = KNearestNeighbours.fit(x_train, y_train, k_neighbours = 5, algo = Algo.KD_TREE, leaf_size = 20)

    y_hat = knn.predict(x_test)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, np.array(list(map(int, x_test_predict))))

    print(cm)
    ```
    """

    @staticmethod
    def fit(x_train, y_train, k_neighbours, algo: Algo = Algo.BRUTE, leaf_size = 20):
        return KNearestNeighbours(x_train, y_train, k_neighbours, algo, leaf_size)

    def __init__(self, x_train, y_train, k_neighbours, algo, leaf_size) -> None:
        self.algo: Algo = algo
        if algo == Algo.BRUTE:
            self.x_train = x_train
            self.y_train = y_train
        elif algo == Algo.KD_TREE:
            self.leaf_size = leaf_size
            self.__get_kdtree(x_train, y_train)
        self.k_neighbours = k_neighbours
    
    def __get_kdtree(self, x_train: np.ndarray, y_train: np.ndarray):
        index_var: np.ndarray = np.arange(x_train.shape[1]) # number of columns (features)
        # sort index by variance
        order = np.argsort(x_train.var(0))
        self.index_var = index_var[order][::-1] # Descending order
        col = 1
        self.tree, childs_x, childs_y = KNearestNeighbours.__split_tree_node(x_train, y_train, self.index_var[0])
        current_nodes = [self.tree]
        nb_feature = x_train.shape[1]
        while True:
            new_child_x = []
            new_child_y = []
            new_nodes = []
            for i in range(len(current_nodes)):
                node_left, child_x_left, child_y_left = KNearestNeighbours.__split_tree_node(childs_x[2 * i], childs_y[2 * i], self.index_var[col % nb_feature])
                node_right, child_x_right, child_y_right = KNearestNeighbours.__split_tree_node(childs_x[2 * i + 1], childs_y[2 * i + 1], self.index_var[col % nb_feature])
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

    def __get_distance(self, point, x_train = None):
        x_train = self.x_train if x_train is None else x_train
        distance = np.ones(x_train.shape) * point
        distance = (distance - x_train) ** 2
        result = np.sqrt(np.sum(distance ** 2, axis=1))
        return np.array(result)

    def __predict_point(self, point):
        if self.algo == Algo.BRUTE:
            order = np.argsort(self.__get_distance(point))
            y_train = self.y_train[order][:self.k_neighbours]
            return sum(y_train) / len(y_train)
        elif self.algo == Algo.KD_TREE:
            node: Node = self.tree
            col = 0
            while not node.isLeaf:
                index = self.index_var[col % len(self.index_var)]
                node = node.left if node.value[index] > point[index] else node.right
                col += 1
            x_train = node.value
            order = np.argsort(self.__get_distance(point, x_train))
            y_train = node.label[order][:self.k_neighbours]
            return sum(y_train) / len(y_train)

    def predict(self, x_test):
        return np.array([self.__predict_point(point) for point in x_test])