import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

##file = 'data/train-images-idx3-ubyte'
##arr = idx2numpy.convert_from_file(file)

X_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
Y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
Y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')

print("X_Train",X_train.shape)
print("Y_Train",Y_train.shape)
print(Y_train[:12])


print("X_Test",X_test.shape)
print("Y_Test",X_test.shape)
print(Y_train[:30])
np.set_printoptions(linewidth=np.nan)

print(X_train[12])
digit = X_train[12]

plt.imshow(digit)
plt.show()

