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
        """
        fit the k-nn algorithm

        Arguments
        -------
        x_train      : 2D numpy array containing the features

        y_train      : 1D array containing the labels

        k_neighbours : first k neighbours to check

        algo         : algorithm to use for data structuring (brute or kd-tree)

        leaf_size    : approximate size of leaf if using the kd-tree algorithm

        Usage
        -----
        ```python
        from k_nearest_neighbours import KNearestNeighbours, Algo

        x_train, y_train, x_test, y_test = ...

        knn = KNearestNeighbours.fit(x_train, y_train, k_neighbours = 5, algo = Algo.KD_TREE, leaf_size = 20)

        y_hat = knn.predict(x_test)

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, np.array(list(map(int, x_test_predict))))

        print(cm)
        ```
        """
        if k_neighbours > leaf_size and algo == Algo.KD_TREE:
            raise ValueError('k value must be less than the leaf size')
        if k_neighbours > len(y_train):
            raise ValueError('k must be less than the total number of training values')

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
        # Create the parent node with the first split
        self.tree, childs_x, childs_y = KNearestNeighbours.__split_tree_node(x_train, y_train, self.index_var[0])
        current_nodes = [self.tree]
        nb_feature = x_train.shape[1]
        while True:
            # variable to replace the current nodes and childs
            new_child_x = []
            new_child_y = []
            new_nodes = []
            for i in range(len(current_nodes)):
                # left and right split for each node
                node_left, child_x_left, child_y_left = KNearestNeighbours.__split_tree_node(childs_x[2 * i], childs_y[2 * i], self.index_var[col % nb_feature])
                node_right, child_x_right, child_y_right = KNearestNeighbours.__split_tree_node(childs_x[2 * i + 1], childs_y[2 * i + 1], self.index_var[col % nb_feature])
                new_child_x += child_x_left + child_x_right
                new_child_y += child_y_left + child_y_right
                new_nodes += [node_left, node_right]
                current_nodes[i].left = node_left
                current_nodes[i].right = node_right
            # replace the local variables
            childs_x = new_child_x
            childs_y = new_child_y
            current_nodes = new_nodes
            # stop before reach leaf_size
            if len(childs_y[0]) / 2 <= self.leaf_size:
                # Create the leaf nodes
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
        # Get the index of the middle el and make it a node
        median = len(x) // 2
        node = Node(x[median], y[median])
        # store the child of the parts before and after the median
        childs_x = [x[:median], x[median + 1:]]
        childs_y = [y[:median], y[median + 1:]]
        return node, childs_x, childs_y

    def __get_distance(self, point, x_train = None):
        # Compute distance between the point and the given (or not) training values
        x_train = self.x_train if x_train is None else x_train
        # create a matrx containing n (number of training data) times the given point
        distance = np.ones(x_train.shape) * point
        # mean squarred error
        distance = (distance - x_train) ** 2
        result = np.sqrt(np.sum(distance, axis=1))
        return np.array(result)

    def __predict_point(self, point):
        # default values
        x_train = None
        y_train = None
        if self.algo == Algo.BRUTE:
            x_train = self.x_train
            y_train = self.y_train
        #get values for kd_tree algo
        if self.algo == Algo.KD_TREE:
            # Search for the corresponding leaf node to the point
            node: Node = self.tree
            col = 0
            while not node.isLeaf:
                index = self.index_var[col % len(self.index_var)]
                node = node.left if node.value[index] > point[index] else node.right
                col += 1
            x_train = node.value
            y_train = node.label
        # order by distance
        order = np.argsort(self.__get_distance(point, x_train))
        y_train = y_train[order][:self.k_neighbours]
        # mean value of the k nearest neighbours (same weight)
        return np.argmax(np.bincount(y_train))

    def predict(self, x_test):
        if type(x_test) == list: x_test = np.array(x_test)
        if type(x_test) != np.ndarray:
            raise TypeError('x_test must be a list or numpy array')
        nbFeature = len(self.tree.value) if self.algo == Algo.KD_TREE else self.x_train.shape[1]
        if x_test.shape[1] != nbFeature:
            raise ValueError('The x_test must have the same number of features given by the training examples array')
        return np.array([self.__predict_point(point) for point in x_test])