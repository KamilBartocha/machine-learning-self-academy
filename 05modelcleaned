import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

# load data
X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')


# delete 0 and 1 from data

def delete_zero_one(X, Y):
    X_0 = np.delete(X, np.where(Y == 0), 0)
    Y_0 = np.delete(Y, np.where(Y == 0))
    X_1 = np.delete(X_0, np.where(Y_0 == 1), 0)
    Y_1 = np.delete(Y_0, np.where(Y_0 == 1))
    return X_1, Y_1


def change_labels(Y):
    # change labels: 1 - prime, 0 - not prime
    for i in range(np.size(Y)):
        if Y[i] == 4 or Y[i] == 6 or Y[i] == 8 or Y[i] == 9:
            Y[i] = 0
        else:
            Y[i] = 1


def accuracy(X, Y, model):
    X = np.reshape(X, (X.shape[0], -1))
    Y_predict = model.predict(X)
    z = 0
    for i in range(np.size(Y)):
        if Y[i] == Y_predict[i]:
            z = z + 1

    accuracy = z / Y_test1.size * 100
    print('accuracy:', accuracy, '%')


def normalize(X):
    return X / 255
# val set cross validation?

# PCA: no need

# print("X_Train", X_train1.shape)

class LogRegressionGM(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        X = np.reshape(X, (X.shape[0], -1))
        random_gen = np.random.RandomState(self.random_state)
        self.weights = random_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (Y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

            # cost
            cost = (-Y.dot(np.log(output)) - ((1 - Y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


X_train1, Y_train1 = delete_zero_one(X_train, Y_train)
X_test1, Y_test1 = delete_zero_one(X_test, Y_test)

change_labels(Y_train1)
change_labels(Y_test1)

X_train1 = normalize(X_train1)
X_test1 = normalize(X_test1)

lrgm = LogRegressionGM(eta=0.05, n_iter=100, random_state=1)
lrgm.fit(X_train1, Y_train1)

accuracy(X_test1, Y_test1, lrgm)
