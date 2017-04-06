import csv
import numpy as np
import sys


def Find_Mean( my_data ) :
    my_mean = np.zeros((106,1))
    temp = np.transpose( my_data )
    for i in range(106) :
       my_mean[i][0] = np.mean( temp[i] )
    return my_mean

data = np.zeros( (32561, 107 ))
dataA = np.zeros( (24720, 107 ))
dataB = np.zeros( (7841, 107 ))

i = 0
with open(sys.argv[3]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        data[i][:106] = row
        i += 1
csvfile.close()

i = 0
with open(sys.argv[4]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    for row in reader :
        data[i][106] = row[0]
        i += 1
csvfile.close()

i = 0
a = 0
b = 0
for i in range(32561) :
    if data[i][106] == 0 :
        dataA[a] = data[i]
        a += 1
    else :
        dataB[b] = data[i]
        b += 1


#////////////////////////////////////////////////////////////////////////////
meanA = np.zeros((106,1))
meanB = np.zeros((106,1))
varianceA = np.zeros((106,106))
varianceB = np.zeros((106,106))
variance = np.zeros((106,106))

meanA = Find_Mean( dataA )
meanB = Find_Mean( dataB )

dataA_T = np.transpose(dataA)
dataB_T = np.transpose(dataB)

varianceA = np.cov(dataA_T[:106])
varianceB = np.cov(dataB_T[:106])
variance = (len(dataA)/32561)*varianceA + (len(dataB)/32561)*varianceB

temp = np.transpose(meanA - meanB)
variance_inverse = np.linalg.inv(variance)
w_T = np.dot(temp,variance_inverse)

meanA_T = np.transpose(meanA)
meanB_T = np.transpose(meanB)

b1 = -0.5*np.dot(np.dot(meanA_T,variance_inverse),meanA)
b2 = 0.5*np.dot(np.dot(meanB_T,variance_inverse),meanB)
b3 = np.log(len(dataA)/len(dataB))
b = (b1 +b2 +b3)
#/////////////////////////////////////////////////////////////////////////////////
test_data = np.zeros( (16281, 106 ))
predict_Ans = [[0]*2]*16281

i = 0
with open(sys.argv[5]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        test_data[i] = row
        i += 1
csvfile.close()

with open(sys.argv[6],'w') as outfile :
    writer = csv.writer(outfile, delimiter=',')
    out = ['id','label']
    writer.writerow(out)
    for j in range(16281) :
        Z = np.sum( np.multiply( test_data[j], w_T ) ) + b
        Sigmoid = 1 / (1  + np.exp( -(Z) ))
        predict_Ans[j][0] = j+1
        predict_Ans[j][1] = ( 0 if Sigmoid >= 0.5 else 1 )
        writer.writerow(predict_Ans[j])

outfile.close()















