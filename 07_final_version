import idx2numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
"""

##### Wnioski i komentarz: #####

Prezentuję dwa podejścia, w pierwszym maksymalizuję zlogarytmizowaną
funkcję wiarygodności, gdzie dla parametrów:
eta = 0.0005, n_iter=101, random_state=1 model osiaga 92.22%

oraz klasyczny model regresji logisrycznej SGD z momentum, który
dla parametrów: eta = 0.0001, epoki = 7, beta = 0.98 osiaga 92,56%

- Standaryzacja danych dale lepsze wyniki niż normalizacja
- Zmniejszanie wymiarowości danych, np za pomocą pca osłabia działanie
  modelu, więc wyciąłem kod
- model osiąga najlepsze wyniki jeśli współczynnik eta bedzie relatywnie
  niski czyli mniej niż tysięczne, za to beta bliskie jedynki
- zwiększanie powtórzeń lub epok nie sprzyja wynikom, model się
  przetrenowuje i osiąga bardzo małe wzrosty celności:
  z 1 na 7 epok dziesiątki procenta 


"""
##### data files are in directory named 'data' #####

# load data
X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

#deleting 'o' and '1' samples from dataset
def delete_zero_one(X, Y):
    X_0 = np.delete(X, np.where(Y == 0), 0)
    Y_0 = np.delete(Y, np.where(Y == 0))
    X_1 = np.delete(X_0, np.where(Y_0 == 1), 0)
    Y_1 = np.delete(Y_0, np.where(Y_0 == 1))
    return X_1, Y_1

#changing labels 'Y' for primes - 1 and not primes - 0
def change_labels(Y):
    for i in range(np.size(Y)):
        if Y[i] == 4 or Y[i] == 6 or Y[i] == 8 or Y[i] == 9:
            Y[i] = 0
        else:
            Y[i] = 1

# simplest normalization, standarize works better
def normalize(X):
    return X / 255

#standardization
def standarize(X, X_test):
    X = np.reshape(X, (X.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)
    return X, X_test

# PCA: no need
"""


 ##### FIRST TRY ######

class ModelLR1(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        #X = np.reshape(X, (X.shape[0], -1)) 
        random_gen = np.random.RandomState(self.random_state)
        self.weights = random_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            input = self.input(X)
            output = self.activation(input)
            errors = (Y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

            # cost
            cost = (-Y.dot(np.log(output)) - ((1 - Y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.input(X)) >= 0.5, 1, 0)


# print accurancy of a model only for FIRST TRY
def accuracy(X, Y, model):
    X = np.reshape(X, (X.shape[0], -1))
    Y_predict = model.predict(X)
    z = 0
    for i in range(np.size(Y)):
        if Y[i] == Y_predict[i]:
            z = z + 1

    accuracy = z / Y_test1.size * 100
    print('accuracy:', accuracy, '%')
####MAIN####

lr1 = ModelLR1(0.0005, 101, 1)
lr1.fit(X_train1, Y_train1)
accuracy(X_test1, Y_test1, lr1)
"""


##### SECOND TRY ######

class ModelLR2(object):

    def __init__(self, eta=0.05, epoch=1, beta=0.95, random_state=0.5):
        self.eta = eta
        self.epoch = epoch
        self.beta = beta
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        var = X.shape[0]
        self.weights = np.ndarray(shape=(X.shape[1], 1), dtype=float)
        self.weights.fill(0)
        v = self.weights
        x = 0
        self.u = 0

        for j in range(self.epoch):
            for i in range(var):
                input = -X[i].dot(self.weights) - self.u
                output = self.activation(input)
                cache = (output - Y[i]) * (output ** 2) * np.exp(-np.clip(input, -250, 250))
                gradient = cache * X[i].T
                gradient = np.reshape(gradient, self.weights.shape)
                v = self.beta * v + self.eta * gradient
                x = self.beta * x + self.eta * cache
                self.u = self.u - x
                self.weights = self.weights - v

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = np.ndarray(shape=(X.shape[0],), dtype=int)
        for i in range(X.shape[0]):
            s_predict = 1 / (1 + np.exp(-X[i].dot(self.weights) - self.u))
            if s_predict > 0.5:
                Y[i] = 1
            else:
                Y[i] = 0
        return Y

    @staticmethod
    def evaluate(self, Y_test1: np.ndarray, Y_predict: np.ndarray) -> float:
        n_error = 0
        for i in range(Y_predict.shape[0]):
            if Y_predict[i] == Y_test1[i]:
                n_error += 1
        return n_error / Y_predict.shape[0]

X_train1, Y_train1 = delete_zero_one(X_train, Y_train)
X_test1, Y_test1 = delete_zero_one(X_test, Y_test)

change_labels(Y_train1)
change_labels(Y_test1)

#X_train1 = normalize(X_train1)
#X_test1 = normalize(X_test1)

X_train1, X_test1 = standarize(X_train1, X_test1)

lr2 = ModelLR2(0.0001, 7, 0.98, 0.5)
lr2.fit(X_train1, Y_train1)
Y_predict = lr2.predict(X_test1)
print('accuracy:', lr2.evaluate(lr2, Y_test1, Y_predict), '%')

