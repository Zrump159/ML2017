import numpy as np

matrixA = np.loadtxt("matrixA.txt", dtype='i', delimiter=',')

print(matrixA)

matrixB = np.loadtxt("matrixB.txt", dtype='i', delimiter=',')

print(matrixB)

matrixC = np.dot(matrixA , matrixB)

print(matrixC)

np.sort(matrixC)

print(matrixC)

np.savetxt('ans.txt', matrixC, fmt='%s',delimiter='\n')

