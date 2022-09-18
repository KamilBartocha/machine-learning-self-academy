import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

print(Y_train[:30])

#delete 0 and 1 from train
X_train0 = np.delete(X_train, np.where(Y_train == 0), 0)
Y_train0 = np.delete(Y_train, np.where(Y_train == 0))
print(Y_train0[:30])

X_train1 = np.delete(X_train0, np.where(Y_train0 == 1), 0)
Y_train1 = np.delete(Y_train0, np.where(Y_train0 == 1))
print(Y_train1[:30])

#delete 0 and 1 from test

X_test0 = np.delete(X_test, np.where(Y_train == 0), 0)
Y_test0 = np.delete(Y_test, np.where(Y_train == 0))

X_test1 = np.delete(X_test0, np.where(Y_train0 == 1), 0)
Y_test1 = np.delete(Y_test0, np.where(Y_train0 == 1))

for i in range(np.size(Y_train1)):
    if Y_train1[i] == 4 or Y_train1[i] == 6 or Y_train1[i] == 8 or Y_train1[i] == 9:
        Y_train1[i] = 0
    else:
        Y_train1[i] = 1

print(Y_train1[:30])
print("X_Train",X_train1.shape)
print("Y_Train",Y_train1.shape)
np.set_printoptions(linewidth=np.nan)
