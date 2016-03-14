import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
std = 1

def printMatrix(M):
	for row in M:
		for val in row:
			print '{:4}'.format(val),
		print
	print '\n'

def euclideanDistance(v1, v2):
	return LA.norm(v1-v2)

def getAffinityMatrix(data_vect):
	size = len(data_vect)
	A = [[0 for x in range(size)] for x in range(size)]
	
	dist = 0
	
	for i in range(size):
		for j in range(size):
			dist = euclideanDistance(data_vect[i], data_vect[j])
			dist = -(dist/(2*(std**2)))
			dist = np.exp(dist)
			A[i][j] = dist

	return A

def getDiagonalMatrix(A):
	dim = np.shape(A)
	numrows = dim[0] #len(A)
	numcols = dim[1] #len(A[0])
	D = [[0 for x in range(numrows)] for x in range(numcols)]
	
	d_sum = 0

	for i in range(numrows):
		d_sum = np.sum(A[i])
		D[i][i] = d_sum
	
	return D

def getLMatrix(A,D):
	L = np.dot(np.dot(SLA.sqrtm(SLA.inv(D)), A),  SLA.sqrtm(SLA.inv(D)))	
	return L

def eigenvectorsMatrix(L, k):
	eigenvalues, eigenvectors = LA.eigh(L)
	return (eigenvalues[0:k], eigenvectors[:,0:k])

def rowsNormalize(X):
	normalization_vect = np.sum(X, axis=1)
	return X/normalization_vect[:,None]

 

