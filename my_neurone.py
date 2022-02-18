import numpy as np
from util import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

my_matrix = np.array([1,2,3])
create_first_matrix(my_matrix)

b = my_new_array()
print(np.shape(b))

a1 = np.array([[1,2], [1,2]])
a2 = np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]])
a3 = np.array([[1,2], [1,2], [1,2], [1,2], [1,2]])
check_random_matrix(a1,a2,a3)

a4 = np.matmul(a3, a1)
check_mul(a4)

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()
