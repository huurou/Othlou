import os.path

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, Input, concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def readkifu(kifu_list_file):
    print(f'{kifu_list_file}を読み込んでいます')
    banlist = np.empty((0, 2, 8, 8))
    tebanlist = np.empty((0, 1))
    resultlist = np.empty((0, 1))
    banlists = np.empty((0, 2, 8, 8))
    tebanlists = np.empty((0, 1))
    resultlists = np.empty((0, 1))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            k = np.load(kifu)
            for i in k:
                b_ban = i[:64].reshape((8, 8))
                w_ban = i[64:128].reshape((8, 8))
                ban = np.stack([b_ban, w_ban]).reshape((1, 2, 8, 8))
                teban = i[128].reshape(1, 1)
                result = i[129].reshape(1, 1)
                banlist = np.append(banlist, ban, axis=0)
                tebanlist = np.append(tebanlist, teban, axis=0)
                resultlist = np.append(resultlist, result, axis=0)
            cnt += 1
            if cnt % 20 == 0:
                banlists = np.append(banlists, banlist, axis=0)
                tebanlists = np.append(tebanlists, tebanlist, axis=0)
                resultlists = np.append(resultlists, resultlist, axis=0)
                banlist = np.empty((0, 2, 8, 8))
                tebanlist = np.empty((0, 1))
                resultlist = np.empty((0, 1))
        print(f'{kifu_list_file}を読み込みました')
    return banlists, tebanlists, resultlists

if __name__ == '__main__':
    b_train, t_train, r_train = readkifu('kifulist_train.txt')
    b_test, t_test, r_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_value.h5'):
        model = load_model('model_value.h5')
        print('モデルmodel_value.h5を読み込みました')
    else:
        print('モデルはありません')
        ban_input = Input(shape=(2, 8, 8))
        conv_1 = Conv2D(64, (3, 3), padding='same')(ban_input)
        b = BatchNormalization(axis=1)(conv_1)
        r = Activation('relu')(b)
        conv_2 = Conv2D(64, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_2)
        r = Activation('relu')(b)
        x = Flatten()(r)
        dens_1 = Dense(64, activation = 'relu')(x)
        teban_input = Input(shape=(1,))
        dens_t = Dense(64, activation = 'relu', name='dense_T')(teban_input)
        merged = concatenate([dens_1, dens_t])
        dens_2 = Dense(64, activation='relu')(merged)
        result_output = Dense(1, activation='sigmoid')(dens_2)
        model = Model(inputs=[ban_input, teban_input],
                      outputs=[result_output])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath = 'model_value.h5')
    history = model.fit([b_train, t_train], [r_train],
                        epochs=20, batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stopping, md_ch])

    score = model.evaluate([b_test, t_test], [r_test], verbose=0)
    del model
