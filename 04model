import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

# load data
X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

# delete 0 and 1 from data

X_train0 = np.delete(X_train, np.where(Y_train == 0), 0)
Y_train0 = np.delete(Y_train, np.where(Y_train == 0))

X_train1 = np.delete(X_train0, np.where(Y_train0 == 1), 0)
Y_train1 = np.delete(Y_train0, np.where(Y_train0 == 1))

# delete 0 and 1 from test

X_test0 = np.delete(X_test, np.where(Y_train == 0), 0)
Y_test0 = np.delete(Y_test, np.where(Y_train == 0))

X_test1 = np.delete(X_test0, np.where(Y_train0 == 1), 0)
Y_test1 = np.delete(Y_test0, np.where(Y_train0 == 1))

# change labels for bool 1 - prime, 0 - not prime in train set

for i in range(np.size(Y_train1)):
    if Y_train1[i] == 4 or Y_train1[i] == 6 or Y_train1[i] == 8 or Y_train1[i] == 9:
        Y_train1[i] = 0
    else:
        Y_train1[i] = 1

# change labels for bool 1 - prime, 0 - not prime in test set

for i in range(np.size(Y_test1)):
    if Y_test1[i] == 4 or Y_test1[i] == 6 or Y_test1[i] == 8 or Y_test1[i] == 9:
        Y_test1[i] = 0
    else:
        Y_test1[i] = 1

# val set ? cross validation?

# normalization do i need?
X_train1 = X_train1 / 255
X_test1 = X_test1 / 255

# PCA no need for pca

#print("X_Train", X_train1.shape)

class LogRegressionGM(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        X = np.reshape(X, (X.shape[0], -1))
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (Y - output)
            print (errors)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # cost

            cost = (-Y.dot(np.log(output)) - ((1 - Y).dot(np.log(1 - output))))
            #print(cost)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


lrgd = LogRegressionGM(eta=0.05, n_iter=70, random_state=1)

lrgd.fit(X_train1, Y_train1)
X = np.reshape(X_test1, (X_test1.shape[0], -1))
Y_pred=lrgd.predict(X)
sum = 0

for i in range(np.size(Y_test1)):
    if Y_test1[i] == Y_pred[i]:
        sum=sum+1
acc= sum/Y_test1.size *100

print(Y_test1.size)
print(sum)


print('acc:', acc, '%')
