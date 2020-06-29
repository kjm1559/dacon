import pandas as pd
import os
os.environ['PYTHONHASHSEED']=str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
random.seed(777)
import numpy as np
np.random.seed(777)

import tensorflow as tf
tf.random.set_seed(777)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
# tf.config.gpu.set_per_process_memory_fraction(0.10)
# tf.config.gpu.set_per_process_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


from datetime import datetime, timedelta
from tensorflow import keras

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate, Embedding, Flatten, \
    TimeDistributed, Masking, GRU, Permute, Dropout, Conv1D, Reshape, MaxPooling1D, Conv2D, Activation, GlobalMaxPooling1D, Average
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model 
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras import initializers
import math
from scipy.stats import norm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

target_columns = ['hhb', 'hbo2', 'ca', 'na']
train_columns = train.columns[1:].tolist()
for t in target_columns:
  del train_columns[train_columns.index(t)]
input_dense = train_columns[1:]
cat_col = ['rho']

# interpolate function
def interpolate_func(df, columns):
#   for i in range(35):
#     df.loc[df[input_dense[i]] == 0, input_dense[i + 35]] = 0
#   df[columns].replace(0, np.nan, inplace=True)
  
  quard = df[columns].copy()
  linear = df[columns].copy()
  quard.rename(columns={dd:i for i, dd in enumerate(columns)}, inplace=True)
  quard.interpolate(axis=1, method='quadratic', inplace=True, limit=3, limit_direction='both')
#   quard.interpolate(axis=1, method='polinomial', inplace=True, order=5)
  quard[quard < 0] = np.nan
  quard.interpolate(axis=1, inplace=True)
  quard.rename(columns={i:dd for i, dd in enumerate(columns)}, inplace=True)  
  linear.interpolate(axis=1, inplace=True)
  return (quard[columns]*0.8 + linear[columns]*0.2)
#   return quard[columns]

# dst columns setup
dst_columns = [k for k in train.columns if 'dst' in k]

# interpolate left to right train
train[dst_columns] = interpolate_func(train, dst_columns)[dst_columns]

# interpolate right to left train
train[dst_columns[::-1]] = interpolate_func(train, dst_columns[::-1])[dst_columns[::-1]]

# interpolate left to right test
test[dst_columns] = interpolate_func(test, dst_columns)[dst_columns]

# interpolate right to left test
test[dst_columns[::-1]] = interpolate_func(test, dst_columns[::-1])[dst_columns[::-1]]

# normalize rho data
train['rho'] = (train['rho'] -10) / (25 - 10)
test['rho'] = (test['rho'] -10) / (25 - 10)
# category_dict = {}
# for i, rho in enumerate(train['rho'].unique()):
#   category_dict[rho] = i
# train['rho'] = [category_dict[idata.rho] for idata in train.itertuples()]
# test['rho'] = [category_dict[idata.rho] for idata in test.itertuples()]

print(np.sum(train[dst_columns].isna()))

# translate data
def make_X(df):
    X = {'inputs': df[input_dense].to_numpy()}    
    for i, v in enumerate(cat_col):        
        X[v] = df[v].to_numpy()
    # corr = df.corr()

    mask = X['inputs'][:,:35].copy()
    mask[mask > 0] = 1
    mask[mask <= 0] = 0
    X['max'] = np.max(X['inputs'][:,35:70], axis=1).reshape((len(X['inputs']), 1))
    X['max2'] = np.min(X['inputs'][:,35:70], axis=1).reshape((len(X['inputs']), 1))
    # X['max'] = np.max(train[input_dense[35:70]].values, axis=1).reshape((len(X['inputs']), 1))
    # X['max2'] = np.min(train[input_dense[35:70]].values, axis=1).reshape((len(X['inputs']), 1))
    max_ = np.max(train[input_dense[35:70]].values, axis=1).reshape((len(X['inputs']), 1))
    max2_ = np.min(train[input_dense[35:70]].values, axis=1).reshape((len(X['inputs']), 1))
    
    # X['inputs'][:, 35:70] = (X['inputs'][:, 35:70] - X['max2']) / (X['max'] - X['max2'])
    X['inputs'][:, 35:70] = X['inputs'][:, 35:70]/X['max']
    
    # for i in range(len(X['inputs'])):
    #     # X['inputs'][i][35:70] = ((X['inputs'][i][35:70]) / X['inputs'][i][35:70].max())
    #     X['inputs'][i][35:70] = (((X['inputs'][i][35:70]) - X['inputs'][i][35:70].min()) / (X['inputs'][i][35:70].max() - X['inputs'][i][35:70].min()))
    #     # X['inputs'][i][35:70] = (((X['inputs'][i][35:70]) - X['inputs'][i][35:70].mean()) / (X['inputs'][i][35:70].std()))
    
    X['sum_src'] = np.mean(X['inputs'][:,:35], axis=1).reshape((len(X['inputs']), 1))
    X['sum_dst'] = np.mean(X['inputs'][:,35:70], axis=1).reshape((len(X['inputs']), 1))
    
    # print(X['sum_src'].mean(), X['sum_src'].max())
    # print(X['sum_dst'].mean(), X['sum_dst'].max(), X['inputs'][:,35:70].max())
    print('[inputs max, min]', X['inputs'].max(), X['inputs'].min())
    
    # X['inputs'] = np.clip(X['inputs'], 0, 1)


    scale = [1e8, 1e10, 1e12, 1e14]
    # scale = [1e10, 1e14, 1e17, 1e20]    
    # scale = [10000**2, 15000**2, 20000**2, 25000**2]    
    rho_index = [0, 5/15, 10/15, 1]
    # rho_index = [0, 1, 2, 3]
    for i in range(4):
        # X['inputs'][:,35:70][X['rho'] == rho_index[i]] *= scale[i]
        print('[max]',i, X['max'][X['rho'] == rho_index[i]].max(), X['max'][X['rho'] == rho_index[i]].mean(), X['max'][X['rho'] == rho_index[i]].min())
        X['max'][X['rho'] == rho_index[i]] *= scale[i]
        X['max2'][X['rho'] == rho_index[i]] *= scale[i]
        max_[X['rho'] == rho_index[i]] *= scale[i]
        max2_[X['rho'] == rho_index[i]] *= scale[i]
        print('[max]',i, X['max'][X['rho'] == rho_index[i]].max(), X['max'][X['rho'] == rho_index[i]].mean(), X['max'][X['rho'] == rho_index[i]].min())
    
    
    
    print('[max]', X['max'].max(), X['max2'].max())    
    # X['max'] = np.clip(X['max'], 0, 7)
    # X['max2'] = np.clip(X['max2'], 0, 7)
    # X['max'] = np.clip(X['max'], X['max'].mean() - 3*X['max'].std(), X['max'].mean() + 3*X['max'].std())
    # X['max2'] = np.clip(X['max2'], X['max2'].mean() - 3*X['max2'].std(), X['max2'].mean() + 3*X['max2'].std())
    X['max'] = np.clip(X['max'], max_.mean() - 3*max_.std(), max_.mean() + 3*max_.std())
    X['max2'] = np.clip(X['max2'], max2_.mean() - 3*max2_.std(), max2_.mean() + 3*max2_.std())
    print('[max clip]', X['max'].max(), X['max2'].max())    

    X['inputs'] = X['inputs'][:,:].reshape((X['inputs'].shape[0], 2, 35, 1))
    
    return X

def tweedieloss(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    p=1.7#0.85
    loss1 = K.pow(y_true, 2-p)/((1-p) * (2-p))
    loss2 = y_true * K.pow(y_pred + 1e-1, 1-p)/(1-p)
    loss3 = K.pow(y_pred + 1e-1, 2-p)/(2-p)
    dev = 2 * (loss1 - loss2 + loss3)
    return K.mean(dev)

def weighted_mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred) * [1, 1, 2, 1])

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)    

def network_set(x):    
    for i in [1024, 512, 256, 128]:
        x = Dense(i, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
    return x

# define model
def predict_model(input_size, epochs=200, lr=1e-3):    
    inputs = Input(shape=(2, 35, 1), name='inputs')

    # Embedding input
    rho_input = Input(shape=1, name='rho')
    max_input = Input(shape=1, name='max')
    max2_input = Input(shape=1, name='max2')
    sum_src_input = Input(shape=1, name='sum_src')
    sum_dst_input = Input(shape=1, name='sum_dst')

    # masking_inputs = Masking(mask_value=-1e-2,input_shape=(4, 35, 1))(inputs)
    
    # rho_emb = Flatten()(Embedding(4, 2)(rho_input))
        
    # input_data = Concatenate(-1)([inputs, rho_emb])
    
    # 1d
    # x = Reshape((2, 35))(inputs)
    # x = Permute((2, 1))(x)
    # x = Conv1D(32, 2, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(inputs)
    x = Conv2D(64, (1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    print('[first]', x.shape)
    x = Reshape((x.shape[-2], x.shape[-1]))(x)    
    # x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, 5, strides=2, activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv1D(32, 5, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)    
    x = Conv1D(32, 1, kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    x = Flatten()(x)
    x = Concatenate(-1)([x, rho_input, max_input])#, max2_input])#, sum_src_input, sum_dst_input])#, max2_input])    
           
    x = network_set(x)    

    outputs = Dense(4, activation='linear', name='na', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    input_dic = {
        'inputs': inputs, 'rho': rho_input, 'max': max_input, 'max2': max2_input,
        'sum_src': sum_src_input, 'sum_dst': sum_dst_input,
    }
    
    model = Model(input_dic, outputs)#, name='predict_model')
    
    return model

def predict_model_82(input_size, epochs=200, lr=1e-3):    
    inputs = Input(shape=(2, 35, 1), name='inputs')

    # Embedding input
    rho_input = Input(shape=1, name='rho')
    max_input = Input(shape=1, name='max')
    max2_input = Input(shape=1, name='max2')
    sum_src_input = Input(shape=1, name='sum_src')
    sum_dst_input = Input(shape=1, name='sum_dst')

    # masking_inputs = Masking(mask_value=-1e-2,input_shape=(4, 35, 1))(inputs)
    
    # rho_emb = Flatten()(Embedding(4, 2)(rho_input))
        
    # input_data = Concatenate(-1)([inputs, rho_emb])
    
    # 1d
    # x = Reshape((2, 35))(inputs)
    # x = Permute((2, 1))(x)
    # x = Conv1D(32, 2, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(inputs)
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv2D(64, (1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    print('[first]', x.shape)
    x = Reshape((x.shape[-2], x.shape[-1]))(x)    
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, 5, strides=1, activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv1D(32, 5, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv1D(32, 1, kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    x = Flatten()(x)
    
    x = Concatenate(-1)([x, rho_input, max_input])#, max2_input])#, sum_src_input, sum_dst_input])#, max2_input])    
           
    x = network_set(x)    

    outputs = Dense(4, activation='linear', name='na', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    input_dic = {
        'inputs': inputs, 'rho': rho_input, 'max': max_input, 'max2': max2_input,
        'sum_src': sum_src_input, 'sum_dst': sum_dst_input,
    }
    
    model = Model(input_dic, outputs)#, name='predict_model')
    
    return model

def predict_model_origin(input_size, epochs=200, lr=1e-3):    
    inputs = Input(shape=(2, 35, 1), name='inputs')

    # Embedding input
    rho_input = Input(shape=1, name='rho')
    max_input = Input(shape=1, name='max')
    max2_input = Input(shape=1, name='max2')
    sum_src_input = Input(shape=1, name='sum_src')
    sum_dst_input = Input(shape=1, name='sum_dst')

    # masking_inputs = Masking(mask_value=-1e-2,input_shape=(4, 35, 1))(inputs)
    
    # rho_emb = Flatten()(Embedding(4, 2)(rho_input))
        
    # input_data = Concatenate(-1)([inputs, rho_emb])
    
    # 1d
    # x = Reshape((2, 35))(inputs)
    # x = Permute((2, 1))(x)
    # x = Conv1D(32, 2, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(inputs)
    x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv2D(64, (1, 1), activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    print('[first]', x.shape)
    x = Reshape((x.shape[-2], x.shape[-1]))(x)    
    x = MaxPooling1D(pool_size=2)(x)
    # x = Dropout(0.25)(x)
    # x = Conv1D(64, 5, strides=2, activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    # x = Conv1D(64, 5, strides=2, activation='relu', padding='same', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv1D(32, 5, strides=2, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    x = Conv1D(32, 1, kernel_initializer=initializers.glorot_normal(seed=777))(x)
    print('[second]', x.shape)
    x = Flatten()(x)
    
    x = Concatenate(-1)([x, rho_input, max_input])#, max2_input])#, sum_src_input, sum_dst_input])#, max2_input])    
           
    for i in [1024, 512, 256, 128]:
        x = Dense(i, activation='relu', kernel_initializer=initializers.glorot_normal(seed=777))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x) 
        
    # x = Dense(64, kernel_initializer=initializers.glorot_normal(seed=777))(x)

    outputs = Dense(4, activation='linear', name='na', kernel_initializer=initializers.glorot_normal(seed=777))(x)
    
    input_dic = {
        'inputs': inputs, 'rho': rho_input, 'max': max_input, 'max2': max2_input,
        'sum_src': sum_src_input, 'sum_dst': sum_dst_input,
    }
    
    model = Model(input_dic, outputs)#, name='predict_model')
    
    return model

def step_decay(epoch):
    initial_lrate = 5e-3
    drop = 0.5
    epochs_drop = 22.0
    lrate = initial_lrate * math.pow(drop,  
        math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)


def predict_e_model(input_size, epochs=200, lr=1e-3):    
    inputs = Input(shape=(2, 35, 1), name='inputs')

    # Embedding input
    rho_input = Input(shape=1, name='rho')
    max_input = Input(shape=1, name='max')
    max2_input = Input(shape=1, name='max2')
    sum_src_input = Input(shape=1, name='sum_src')
    sum_dst_input = Input(shape=1, name='sum_dst')
    
    input_dic = {
        'inputs': inputs, 'rho': rho_input, 'max': max_input, 'max2': max2_input,
        'sum_src': sum_src_input, 'sum_dst': sum_dst_input,
    }

    # models = [predict_model(35)] * model_len
    models = [predict_model_origin(35), predict_model_82(35), predict_model_origin(35)]
    # models = [predict_model_82(35), predict_model(35)]
    # models = [predict_model(len(input_dense)), predict_model(len(input_dense))]
    models[0].load_weights('./predict_mae_e1.h5')
    models[1].load_weights('./predict_mae_e2.h5')
    models[2].load_weights('./predict_mae_e3.h5')
    # for i in range(model_len) :
    #     models[i].load_weights('./predict_mae' + str(i) + '.h5')
    

    i = 0
    for model in models:
        for layer in model.layers:
            layer.trainable = False
            # layer.name = 'ensemble_' + str(i) + layer.name
        i += 1
        
    ensemble_outputs = [model(input_dic) for model in models]
    # merge = Concatenate(-1)(ensemble_outputs) 
    outputs = Average()(ensemble_outputs)

    model = Model(inputs=input_dic, outputs=outputs)
    # model.compile(optimizer=RMSprop(lr=lr), loss='mae')
    return model
        


# target_num = 10
# loss_list = []
# # for i in [0, 1, 2, 6, 8, 14]:
# for i in range(target_num):
#     print('[train start] :', i)
#     tf.random.set_seed(777 + i)
#     model = predict_model(35)#len(input_dense))
#     # # model.compile(optimizer=Adam(lr=1e-4), loss=[tweedieloss, tweedieloss, tweedieloss, tweedieloss], metrics=['mae'])
#     # # model.compile(optimizer=RMSprop(lr=1e-3), loss='mae', metrics=['mae'])
#     model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
#     model_path = './predict_mae' + str(i) + '.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
#     cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
#     early_stopping = EarlyStopping(patience=50)
#     train_data = make_X(train)
#     train_data['inputs'].reshape(list(train_data['inputs'].shape) + [1])
#     # history = model.fit(train_data, {'hhb':train[target_columns[0]].values, 'hbo2':train[target_columns[1]].values, 'ca':train[target_columns[2]].values, 'na':train[target_columns[3]].values}, batch_size=256, epochs=400, verbose=1, shuffle=True,
#     #                                 validation_split=0.2,
#     #                                 callbacks=[cb_checkpoint, early_stopping])    
#     history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
#                                     validation_split=0.2,
#                                     callbacks=[cb_checkpoint, early_stopping, lrate])
#     print(min(history.history['val_loss']))
#     loss_list.append(min(history.history['val_loss']))
#     model.load_weights('./predict_mae' + str(i) + '.h5')
    
#     result = model.predict(make_X(test))
#     for i, t in enumerate(target_columns):
#         sub[t] += result[:, i]
        
# print(loss_list)
# sub[target_columns] = sub[target_columns] / target_num
# sub.to_csv('submission.csv', index=False)
# print(sub)

train_data = make_X(train)
train_data['inputs'].reshape(list(train_data['inputs'].shape) + [1]) 

# # 1 train
# # [0.8165435194969177, 0.8248805999755859, 0.8308548331260681, 0.8353477716445923, 0.8498175740242004, 0.8372244238853455, 0.8355668783187866, 0.8236991167068481, 0.8369251489639282, 0.8309692144393921]
# tf.random.set_seed(777)
# model = predict_model(35)#len(input_dense))
# model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
# model_path = './predict_mae_e1.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
# cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(patience=50)
# history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
#                                 validation_split=0.2,
#                                 callbacks=[cb_checkpoint, early_stopping, lrate])

# 2 train
# [0.8203421235084534, 0.8016051650047302, 0.8241068720817566, 0.8133254647254944, 0.8363664746284485, 0.8187965750694275, 0.8185847401618958, 0.8352689743041992, 0.8546475172042847, 0.818452775478363]
tf.random.set_seed(777 + 1)
model = predict_model_82(35)#len(input_dense))
model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
model_path = './predict_mae_e2.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=50)
history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping, lrate])

# 3 train
# [0.8011473417282104, 0.8182101845741272, 0.8128118515014648, 0.8116616010665894, 0.8080560564994812, 0.810356080532074, 0.8163599967956543, 0.8267065286636353, 0.8264393210411072, 0.8347173929214478]
tf.random.set_seed(777)
model = predict_model_origin(35)#len(input_dense))
model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
model_path = './predict_mae_e3.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=50)
history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping, lrate])

# 4 train
# [0.8011473417282104, 0.8182101845741272, 0.8128118515014648, 0.8116616010665894, 0.8080560564994812, 0.810356080532074, 0.8163599967956543, 0.8267065286636353, 0.8264393210411072, 0.8347173929214478]
tf.random.set_seed(777 + 4)
model = predict_model_origin(35)#len(input_dense))
model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
model_path = './predict_mae_e1.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=50)
history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping, lrate])

e_model = predict_e_model(35)
result = e_model.predict(make_X(test))
for i, t in enumerate(target_columns):
    sub[t] += result[:, i]

sub[target_columns] = sub[target_columns]
sub.to_csv('submission.csv', index=False)
print(sub)



#--------------------- select random seed ----------------------------------------
# target_num = 10
# loss_list = []
# # for i in [0, 1, 2, 6, 8, 14]:
# for i in range(target_num):
#     print('[train start] :', i)
#     tf.random.set_seed(777 + i)
#     model = predict_model_origin(35)#len(input_dense))
#     # # model.compile(optimizer=Adam(lr=1e-4), loss=[tweedieloss, tweedieloss, tweedieloss, tweedieloss], metrics=['mae'])
#     # # model.compile(optimizer=RMSprop(lr=1e-3), loss='mae', metrics=['mae'])
#     model.compile(optimizer=RMSprop(lr=5e-3), loss='mae', metrics=['mae'])
#     model_path = './predict_mae' + str(i) + '.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
#     cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
#     early_stopping = EarlyStopping(patience=50)
#     train_data = make_X(train)
#     train_data['inputs'].reshape(list(train_data['inputs'].shape) + [1])
#     # history = model.fit(train_data, {'hhb':train[target_columns[0]].values, 'hbo2':train[target_columns[1]].values, 'ca':train[target_columns[2]].values, 'na':train[target_columns[3]].values}, batch_size=256, epochs=400, verbose=1, shuffle=True,
#     #                                 validation_split=0.2,
#     #                                 callbacks=[cb_checkpoint, early_stopping])    
#     history = model.fit(train_data, train[target_columns[:]].values, batch_size=128, epochs=400, verbose=1, shuffle=True,
#                                     validation_split=0.2,
#                                     callbacks=[cb_checkpoint, early_stopping, lrate])
#     print(min(history.history['val_loss']))
#     loss_list.append(min(history.history['val_loss']))
#     model.load_weights('./predict_mae' + str(i) + '.h5')
    
#     # result = model.predict(make_X(test))
#     # for i, t in enumerate(target_columns):
#     #     sub[t] += result[:, i]
    
# print(loss_list)