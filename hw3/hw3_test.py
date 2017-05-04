import numpy as np
import csv
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model

data = ['0']*28709
Ans = [0]*28709
train_data = [[0]*2304]*28709

##/////////////load test data///////////////////
data = ['0']*7178
test_data = [[0]*2304]*7178

i = 0
with open(sys.argv[1]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        data[i] = row[1]
        i += 1
csvfile.close()

for i in range(7178) :
    test_data[i] = data[i].split()

test_data = np.array(test_data,np.float)
test_data = test_data.reshape((test_data.shape[0],48,48,1))
test_data = test_data/255
#////////////////////////////////////////CNN/////////////////////////////////////////////////////
model = load_model('model_save.h5')

score = model.evaluate(train_data,Ans)
print("\nTrain Acc:", score[1])

result = model.predict(test_data)

out = [0]*7178

out = result.argmax(axis=1)

with open(sys.argv[2], 'w',newline='') as outfile :
    writer = csv.writer(outfile , delimiter=',')
    output = [['id','label']]
    writer.writerows(output)

    for i in range(7178):
        output = [[i, out[i]]]
        writer.writerows(output)

outfile.close()
