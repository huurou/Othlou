import os.path
import pickle

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
    banlist = np.empty((0, 3, 8, 8))
    movelist = np.empty((0,64))
    banlists = np.empty((0, 3, 8, 8))
    movelists = np.empty((0,64))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            k = np.load(kifu)
            for i in k:
                b_ban = i[:64].reshape((8, 8))
                w_ban = i[64:128].reshape((8, 8))
                teban = np.full((8,8),i[128])
                ban = np.stack([b_ban, w_ban, teban]).reshape((1,3,8,8))
                m = int(i[129])
                move = np.eye(64)[m].reshape((1,64))
                banlist = np.append(banlist, ban, axis=0)
                movelist = np.append(movelist, move, axis=0)
            cnt += 1
            if cnt % 50 == 0:
                banlists = np.append(banlists, banlist, axis=0)
                movelists = np.append(movelists, movelist, axis=0)
                banlist = np.empty((0, 3, 8, 8))
                movelist = np.empty((0, 64))
        print(f'{kifu_list_file}を読み込みました')
    return banlists, movelists

if __name__ == '__main__':
    b_train, m_train = readkifu('kifulist_train.txt')
    b_test, m_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_move.h5'):
        model = load_model('model_move.h5')
        print('モデルmodel_move.h5を読み込みました')
    else:
        print('モデルはありません')
        ban_input = Input(shape=(3, 8, 8))
        conv_1 = Conv2D(192, (3, 3), padding='same')(ban_input)
        b = BatchNormalization(axis=1)(conv_1)
        r = Activation('relu')(b)
        conv_2 = Conv2D(64, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_2)
        r = Activation('relu')(b)
        conv_3 = Conv2D(64, (3, 3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_3)
        r = Activation('relu')(b)
        x = Flatten()(r)
        dens_1 = Dense(64, activation = 'relu')(x)
        dens_2 = Dense(64, activation='relu')(dens_1)
        dens_3 = Dense(64, activation='relu')(dens_2)
        move_output = Dense(64, activation = 'softmax')(dens_3)
        model = Model(inputs=ban_input,
                      outputs=move_output)
    model.summary()
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath = 'model_move.h5')
    history = model.fit(b_train, m_train,
                        epochs=20, batch_size=64,
                        verbose=1, validation_data=(b_test, m_test)
                        callbacks=[early_stopping, md_ch])
    model.save('model_move.h5')
    del model