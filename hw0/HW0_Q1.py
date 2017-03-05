import random
import sys
import os
import numpy as np

matrix_f = open ("matrixA.txt" , "r")


matrixA = []
for line in matrix_f :
    line = line.strip()
    #strip removed some character from beginning or end of the string.
    matrixA.append([int(a) for a in line.split(',')])

print("matrixA :")
for i in range(matrixA.__len__()) :
    print(matrixA[i])


matrix_f = open ("matrixB.txt" , "r")


matrixB = []
for line in matrix_f :
    line = line.strip()
    #strip removed some character from beginning or end of the string.
    matrixB.append([int(a) for a in line.split(',')])

print("matrixB :")
for i in range(matrixB.__len__()) :
    print(matrixB[i])

matrixAB = [[ 0 for i in range( len(matrixB[0]) ) ] for j in range( len(matrixA ) )]
#create a Am by Bn array
ans = []
total = 0

print("matrixA*B :")

for Ai in range( len(matrixA) ) : #get matrixA mxn m
    for Bj in range(len(matrixB[0])):  # get matrixB mxn n
        for Aj in range ( len(matrixA[0]) ) : #get matrixA mxn n
            total = total + matrixA[Ai][Aj]*matrixB[Aj][Bj]
        matrixAB[Ai][Bj] = total
        ans.append(total)
        total = 0

for i in range( len(matrixAB) ) :
    print(matrixAB[i])

str( ans.sort() )
print("ans = " , ans )

matrix_f = open("ans.txt" , "w" )
for line in ans :
    print(line)
    matrix_f.write("%s \n" % line)






