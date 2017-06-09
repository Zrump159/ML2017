import numpy as np
import keras
from keras.layers import Dot, Embedding, Reshape, Input, Flatten, Add
from keras.models import Sequential, Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel,BiasModel


## data path
user_path = 'users.csv'
movie_path = 'movies.csv'
train_path = 'train.csv'
test_path = 'test.csv'
out_path = 'out.csv'
## parameter
k_factors = 215

def switch(x) :
    return int(np.floor(x/15))

def multicalss( tags, list ) :
    tag_len = len(list)
    data_len = len(tags)
    tags_list = np.zeros((data_len,tag_len))
    for i in range(data_len) :
        for t in tags[i] :
            tags_list[i][list.index(t)] = 1
    return np.array(tags_list,np.int)


def read(path,type) :

    data = []
    tags = []
    tag_list = []

    with open(path,'r',encoding = 'ISO-8859-1') as f :
        f.readline()
        for line in f :
            if type == 'user' :
                data.append(line[:line.find('-')].split('::'))
                data[-1][1] = 1 if data[-1][1] == 'F' else 0
                data[-1][2] = switch(int(data[-1][2]))
            elif type == 'movie' :
                data.append(line[:-1].split('::')[:-1])
                data[-1][1] = data[-1][1][-5:-1]
                tags.append(line[:-1].split('::')[-1].split('|'))
                for t in tags[-1] :
                    if t not in tag_list :
                        tag_list.append(t)
            elif type == 'train' :
                data.append(line[:-1].split(','))
            else :
                data.append(line[:-1].split(','))


    f.close()

    if type == 'movie' :
        tags = multicalss(tags,tag_list)
        data = np.concatenate((data,tags),axis=1)



    return np.array(data,np.int)

def merge_data( user, movie, data ):
    u = []
    m = []

    tmp1 = []
    tmp2 = []

    for line in user :
        tmp1.append(line[0])
    for line in movie :
        tmp2.append(line[0])

    for line in data :
        u.append(user[tmp1.index(line[1])])
        m.append(movie[tmp2.index(line[2])])

    return np.array(u,int),np.array(m,int)

def prepare(d) :
    u = []
    m = []
    r = []
    for l in d :
        u.append(l[1]-1)
        m.append(l[2]-1)
        r.append(l[3])
    return np.array(u,np.int),np.array(m,np.int),np.array(r,np.int)

def shuffle(a,b,c) :
    indices = np.arange(len(a))
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]
    c = c[indices]

    return np.array(a,np.int64),np.array(b,np.int64),np.array(c,np.int64)

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


##----------------------main-----------------------------------

def main() :
    #reading data
    print('Loading data...')
    print('Loading user list...')
    user = read(user_path,'user')
    print(len(user),'user load')
    print('Loading movie list...')
    movie = read(movie_path,'movie')
    print(len(movie), 'movie load')
    print('Loading train data...')
    train_data = read(train_path,'train')
    print(len(train_data), 'train data load...')
    print('Loading test data...')
    test_data = read(test_path,'test')
    print(len(test_data), 'test data load...')

    #merge data
    '''
    print('merge data...')
    train_user, train_movie = merge_data(user,movie,train_data)
    print('train data merge complete')
    '''
    #prepare data
    print('prepare data...')
    u_list,m_list,r_list = prepare(train_data)

    #shuffle data
    print('shuffle data...')
    u_list, m_list, r_list = shuffle(u_list,m_list,r_list)


    #create model
    print('create model...')
    print("r_list shape:", r_list.shape)
    print("u_list shape:",u_list.shape)
    print("m_list shape:", m_list.shape)
    max_u_id = np.max(u_list)+1
    max_m_id = np.max(m_list)+1
    print('k_factors:',k_factors)
    print('max user id:', max_u_id)
    print('max movie id', max_m_id)

    '''
        model = build_model(max_u_id,max_m_id,k_factors)
        model.compile(loss='mse', optimizer='sgd')
        model.summary()
         '''
    #model = BiasModel(max_u_id, max_m_id, k_factors)
    model = CFModel(max_u_id, max_m_id, k_factors)
    model.compile(loss='mse', optimizer='adamax')
    model.summary()

    callbacks = [EarlyStopping('val_loss', patience=4),
                 ModelCheckpoint('my_weight_bias_175.h', save_best_only=True)]
    #history = model.fit([u_list, m_list, u_list, m_list], r_list, epochs=30, validation_split=.1, verbose=2, callbacks=callbacks)
    history = model.fit([u_list, m_list], r_list, epochs=30, validation_split=.1, verbose=2, callbacks=callbacks)




if __name__ == '__main__':
    main()