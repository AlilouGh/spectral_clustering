import numpy as np
from numpy import linalg as LA
import matrixOperations as mo

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

#A = np.diag((1, 2, 3))
#mo.printMatrix(A)

data_vect = np.array([[1, -1, 0], [1, 2, 3], [0, 2, 1]])
mo.printMatrix(data_vect)
A = mo.getAffinityMatrix(data_vect)
mo.printMatrix(A)
D = mo.getDiagonalMatrix(A)
mo.printMatrix(D)
L = mo.getLMatrix(A,D)
mo.printMatrix(L)
v, X = mo.eigenvectorsMatrix(L, 3)
mo.printMatrix(X)
for val in v:
	print '{:4}'.format(val),
print '\n'
Y = mo.rowsNormalize(X)
mo.printMatrix(Y)

name = 'k_means_3'
estimator =  KMeans(n_clusters=3)
fignum = 1

fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
estimator.fit(Y)
labels = estimator.labels_

ax.scatter(Y[:, 1], Y[:, 0], Y[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(Y[y == label, 3].mean(),
              Y[y == label, 0].mean() + 1.5,
              Y[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(Y[:, 3], Y[:, 0], Y[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()
ax.set_zlabel('Petal length')
plt.show()
