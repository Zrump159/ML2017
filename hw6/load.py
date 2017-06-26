import math
import pandas as pd
from keras.layers import Dot, Embedding, Reshape, Input, Flatten, Add
import keras
import sys
#import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
import csv
from CFModel import CFModel,DeepModel,BiasModel
import os

MODEL_WEIGHTS_FILE = 'my_weight_a_215.h'
Outfile = sys.argv[2]
model_type = 'a'
K_FACTORS = 215
RNG_SEED = 1446557


def build_model(users,items,latent_dim = 120 ):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(users,latent_dim,embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(users,1,embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    r_hat = Add()([r_hat, item_bias, user_bias])
    model = keras.models.Model([user_input,item_input], r_hat)

    return model
###############

if model_type == 'b' :
    trained_model = build_model(6040, 3952, 240)
    trained_model.load_weights(MODEL_WEIGHTS_FILE)
    def predict_rating(userid, movieid):
        return trained_model.predict([np.array([userid - 1]), np.array([movieid - 1])])[0][0]

elif model_type == 'd' :
    trained_model = DeepModel(6040, 3952, K_FACTORS)
    trained_model.load_weights(MODEL_WEIGHTS_FILE)

    def predict_rating(userid, movieid):
        return trained_model.rate(userid - 1, movieid - 1)

elif model_type == 'b2' :
    trained_model = BiasModel(6040, 3952, K_FACTORS)
    trained_model.load_weights(MODEL_WEIGHTS_FILE)

    def predict_rating(userid, movieid):
        return trained_model.rate(userid - 1, movieid - 1)

else :
    trained_model = CFModel(6040, 3952, K_FACTORS)
    trained_model.load_weights(MODEL_WEIGHTS_FILE)

    def predict_rating(userid, movieid):
        return trained_model.rate(userid - 1, movieid - 1)



temp_set=[[]]*100336
with open(sys.argv[1]) as csvfileY:
    readerY = csv.reader(csvfileY, delimiter= ',')
    next(readerY, None)
    temp_number_data = 0

    for row in readerY:
        temp_set[temp_number_data] = temp_set[temp_number_data] + row
        temp_number_data += 1
csvfileY.close()

answer = []
for i in range(100336):
    answer.append(predict_rating(int(temp_set[i][1]), int(temp_set[i][2])))


with open(Outfile,'w',newline ="") as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(["TestDataID","Rating"])

    for i in range(100336):
        a=[str(i+1),str(answer[i])]
        ansFile.writerow(a)
csvfile.close()
