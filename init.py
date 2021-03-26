import numpy as np
from sklearn.model_selection import train_test_split

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

def split_dataset(data, output, test_size = .3, validation_size = .5):
    # https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=test_size)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=validation_size)
    return x_train, y_train, x_test, y_test#, x_val, y_val

def error(y_true, y_pred):
    return sum((y_true - y_pred) ** 2) ** 1/2

def reduce(x):
    """
    centre et réduit les données en fonction de leur variance
    """
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x = np.substr
    return 

class KNearestNeighbours:

    @staticmethod
    def fit(x_train, y_train, k_neighbours):
        return KNearestNeighbours(x_train, y_train, k_neighbours)

    def __init__(self, x_train, y_train, k_neighbours) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.k_neighbours = k_neighbours

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