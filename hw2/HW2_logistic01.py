import csv
import numpy as np
import sys


def Normalization( my_data ) :
    my_data = np.transpose( my_data )
    for i in range(105) :
        mean = np.mean( my_data[i] )
        variance = np.std( my_data[i] ) if np.std( my_data[i] ) != 0 else 1
        my_data[i] = np.divide( np.subtract(my_data[i], mean), variance )
    my_data = np.transpose(my_data)
    return my_data

data = np.zeros( (32561, 105 ))
Answer = np.zeros(32561)

i = 0
with open(sys.argv[3]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        data[i][0] = row[0]
        data[i][1:105] = row[2:106]
        i += 1
csvfile.close()

i = 0
with open(sys.argv[4]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    for row in reader :
        Answer[i] = row[0]
        i += 1
csvfile.close()
#/////////////////////////////////////////////////////////
with open('basic_weight.csv') as csvfile :
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader :
        weight = row[0:105]
        bias = row[105]
weight = np.array(weight,np.float)
bias = np.float(bias)
csvfile.close()
#///////////////////////////////////////////////////////////
data = Normalization(data)

set = 5000
set_bias = 5000
#weight = np.zeros(105)
#bias = 0.0
g_weight = np.zeros(105)
g_bias = 0.0
Lrnrate = 0.0001
itr = 500
trainset = np.zeros(( set, 105 ))
trainset = data[set_bias:set+set_bias,...]
z = np.zeros(set)
sigmoid = np.zeros(set)
ln_fuc = np.zeros(set)
error = 0


for i in range(itr) :
    Loss_fuc = 0.0
    g_weight = np.zeros(105)
    g_bias = 0.0
    error = 0
    for j in range(set) :
        z[j] = np.sum( np.multiply( trainset[j], weight ) ) + bias
        sigmoid[j] = 1. / (1.  + np.exp( -z[j] ))
        ln_fuc = np.log(sigmoid[j]) if Answer[set_bias+j] == 1 else np.log( 1 - sigmoid[j] )
        Ans_minus_sigomid = np.subtract( Answer[set_bias+j], sigmoid[j] )
        g_bias += -Ans_minus_sigomid
        g_weight += -np.multiply( trainset[j], Ans_minus_sigomid )
        Loss_fuc += -ln_fuc
        error += 0 if Answer[set_bias+j] == np.around(sigmoid[j]) else 1
    print('iteration : ',i)
    print('Error : ', error)
    print('Loss : ',Loss_fuc)
    weight += -(Lrnrate*g_weight)
    bias += -(Lrnrate*g_bias)

#//////////////////////////////////////////////////////////
temp = [0.0]*106
with open("basic_weight.csv", 'w' ) as outfile :
    writer = csv.writer(outfile, delimiter=',')
    temp[:105] = weight
    temp[105] = bias
    writer.writerow(temp)
csvfile.close()
#/////////////////////////////////////////////////////////
test_data = np.zeros( (16281, 105 ))
predict_Ans = [[0]*2]*16281

i = 0
with open(sys.argv[5]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        test_data[i][0] = row[0]
        test_data[i][1:105] = row[2:106]
        i += 1
csvfile.close()

test_data = Normalization(test_data)

with open(sys.argv[6], 'w' ) as outfile :
    writer = csv.writer(outfile, delimiter=',')

    out = ['id','label']
    writer.writerow(out)
    for j in range(16281) :
        Z = np.sum( np.multiply( test_data[j], weight ) ) + bias
        Sigmoid = 1 / (1  + np.exp( -Z ))
        predict_Ans[j][0] = j+1
        predict_Ans[j][1] = ( 1 if Sigmoid >= 0.5 else 0 )
        writer.writerow(predict_Ans[j])

outfile.close()
