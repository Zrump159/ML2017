import numpy as np
import string
import sys
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import theano
from keras.models import model_from_json
from keras.layers import LSTM
import codecs
from keras.models import load_model

import os
os.environ['THEANO_FLAGS'] = "device=gpu0"

#####################
###   parameter   ###
#####################
split_ratio = 0.3
embedding_dim = 100
nb_epoch = 100
batch_size =128


################
###   Util   ###
################
def read_data(path, training):
    print('Reading data from ', path)
    with open(path, 'r', encoding = 'utf-8-sig') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            if training:
                start = line.find('\"')
                end = line.find('\"', start + 1)
                tag = line[start + 1:end].split(' ')
                article = line[end + 2:]

                for t in tag:
                    if t not in tags_list:
                        tags_list.append(t)

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start + 1:]

            articles.append(article)

        if training:
            assert len(tags_list) == 38, (len(tags_list))
            assert len(tags) == len(articles)
    return (tags, articles, tags_list)


def get_embedding_dict(path):
    embedding_dict = {}
    with open(path, 'r', encoding = 'utf-8-sig') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict


def get_embedding_matrix(word_index, embedding_dict, num_words, embedding_dim):
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

###########################
###   custom metrices   ###
###########################
def f1_score(y_true, y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred, thresh), dtype='float32')
    tp = K.sum(y_true * y_pred, axis=-1)

    precision = tp / (K.sum(y_pred, axis=-1) + K.epsilon())
    recall = tp / (K.sum(y_true, axis=-1) + K.epsilon())
    return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))


#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (_, X_test, _) = read_data(sys.argv[1], False)
    tag_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE",
                'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL',
                'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL',
                'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR',
                'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER',
                'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION',
                'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']

    ### tokenizer for all data
    import json
    tokenizer = Tokenizer()
    with open('json_RNN2.txt', 'r') as fp:
        tokenizer.word_index = json.load(fp)
    fp.close()
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    print('Convert to index sequences.')
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print('Padding sequences.')
    max_article_length = 306
    test_sequences = pad_sequences(test_sequences, maxlen=max_article_length)


    ### build model
    print('Building model.')
    model = load_model("my_model(RNN2).h5",
                       custom_objects={'f1_score':f1_score})

    Y_pred = model.predict(test_sequences)
    thresh = 0.4

    for pred in Y_pred:
        no_out = 0
        for v in pred :
            no_out += 0 if v < thresh else v
        if (no_out == 0 ):
            pred +=  0.1
    account = {"0":0,"1":0,"2":0,"3":0,"other":0}

    with open(sys.argv[2], 'w') as output:
        print('\"id\",\"tags\"', file=output)
        for pred in Y_pred :
            if np.max(pred) < thresh :
                pred[np.argmax(pred)] += thresh
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        # = ["0":]
        for index, labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i, value in enumerate(labels) if value == 1]
            labels_original = ' '.join(labels)
            print('\"%d\",\"%s\"' % (index, labels_original), file=output)


if __name__ == '__main__':
    main()
