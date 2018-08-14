from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras import initializers
import random
import pandas as pd
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def sigmoid_10k(x):
    return 10000*K.sigmoid(x)

def Mul(input_dim):
    inputs = Input(shape=(input_dim, ))

    out = Dense(10, kernel_initializer='random_normal', bias_initializer='zeros')(inputs)
    out = Activation('relu')(out)

    out = Dense(10, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = Activation('relu')(out)

    out = Dense(10, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = Activation('relu')(out)

    out = Dense(10, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = Activation('relu')(out)

    out = Dense(8, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = Activation('relu')(out)

    out = Dense(1, kernel_initializer='random_normal', bias_initializer='zeros')(out)
   
    model = Model(inputs = inputs, outputs = out)
    return model

def gen_data(num, dst=None):

    X1 = np.linspace(-1, 1, num)
    X2 = np.linspace(-1, 1, num)
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    Y = X1*X2 # + np.random.normal(0, 0.05, (num, )) #生成Y并添加噪声

    data = pd.DataFrame(columns=['Y','X1','X2'])
    data['Y'] = Y
    data['X1'] = X1
    data['X2'] = X2

    if dst is not None:
        data.to_csv(dst, index=None)

    return data

def train():
    WORK_DIR = '../workdir-'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
    TARGET_DIR = os.path.join(WORK_DIR, 'target')
    MODEL_DIR = os.path.join(WORK_DIR, 'model')
    LOG_DIR = os.path.join(WORK_DIR, 'log')
    BATCH_SIZE = 64

    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    mul_data = gen_data(10000, os.path.join(OUTPUT_DIR, 'mul.csv'))

    label = mul_data['Y'].as_matrix()
    data = mul_data[['X1','X2']].as_matrix()

    mul_test_data = gen_data(1000, os.path.join(OUTPUT_DIR, 'mul_test.csv'))
    test_label = mul_test_data['Y'].as_matrix()
    test_data = mul_test_data[['X1','X2']].as_matrix()

    '''
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for idx_train, idx_val in skf.split(label,label):
        train_data = data[idx_train]
        train_label = label[idx_train]
        val_data = data[idx_val]
        val_label = label[idx_val]
    '''
    idx_val = random.sample(list(range(10000)), 2000)
    idx_train = list(set(range(10000)).difference(idx_val))
    print("idx_val : {}".format(len(idx_val)))
    print("idx_train:{}".format(len(idx_train)))
    train_data = data[idx_train]
    train_label = label[idx_train]
    val_data = data[idx_val]
    val_label = label[idx_val]

    model = Mul(2)
    model.compile(loss='mse', optimizer='sgd')
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'model.{epoch:02d}-{val_loss:.4f}.hdf5'), period=1)   #'model.{epoch:02d-{val_loss:.2f}}.hdf5'
    tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_grads=True, write_graph=False,write_images=True)
    
    model.fit(train_data, train_label, validation_data=(val_data,val_label),\
                batch_size=BATCH_SIZE, epochs=30, shuffle=True, \
                callbacks=[model_checkpoint, tensorboard, early_stopping],\
                verbose=1)
    
    preds = model.predict(test_data, batch_size=BATCH_SIZE, verbose=1)
    err_score = mean_squared_error(np.array(test_label), np.array(preds))
    print("mean_squared_error:{}".format(err_score))

    mul_test_data['pred'] = list(preds)
    mul_test_data.to_csv(os.path.join(TARGET_DIR, 'predict.csv'), index=None)\

if __name__ == '__main__':
    train()