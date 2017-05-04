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

##//////////////////////////////////
i = 0
with open(sys.argv[1]) as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        data[i] = row[1]
        Ans[i] = row[0]
        i += 1
csvfile.close()

for i in range(28709) :
    train_data[i] = data[i].split()

train_data = np.array(train_data,np.float)
Ans = np.array(Ans,np.float)
Ans = np_utils.to_categorical(Ans, 7)
train_data = train_data.reshape((train_data.shape[0],48,48,1))
train_data = train_data/255
##/////////////define network///////////////////

model2 = Sequential()
model2.add(Conv2D(25,(4,4),input_shape=(48,48,1)))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(50,(3,3)))
model2.add(MaxPooling2D((3,3)))
model2.add(Flatten())
model2.add(Dense(units=120,activation='relu'))
model2.add(Dense(units=80,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(train_data,Ans,batch_size=100,epochs=10)

score = model2.evaluate(train_data,Ans)
print("\nTrain Acc:", score[1])

model2.save('model_save.h5')
