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
    banlist = np.empty((0, 2, 8, 8))
    tebanlist = np.empty((0, 1))
    movelist = np.empty((0,64))
    banlists = np.empty((0, 2, 8, 8))
    tebanlists = np.empty((0, 1))
    movelists = np.empty((0,64))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            k = np.load(kifu)
            for i in k:
                b_ban = i[:64].reshape((8, 8))
                w_ban = i[64:128].reshape((8, 8))
                ban = np.stack([b_ban, w_ban]).reshape((1,2,8,8))
                teban = i[128].reshape((1,1))
                move = i[129:].reshape((1,64))
                banlist = np.append(banlist, ban, axis=0)
                tebanlist = np.append(tebanlist, teban, axis=0)
                movelist = np.append(movelist, move, axis=0)
            cnt += 1
            if cnt % 50 == 0:
                banlists = np.append(banlists, banlist, axis=0)
                tebanlists = np.append(tebanlists, tebanlist, axis=0)
                movelists = np.append(movelists, movelist, axis=0)
                banlist = np.empty((0, 2, 8, 8))
                tebanlist = np.empty((0, 1))
                movelist = np.empty((0,64))
        movelists_split = []
        movelists_split = np.split(movelists, 64, axis = 1)
        print(f'{kifu_list_file}を読み込みました')
    return banlists, tebanlists, movelists_split

if __name__ == '__main__':
    b_train, t_train, m_train = readkifu('kifulist_train.txt')
    b_test, t_test, m_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_move_64.h5'):
        model = load_model('model_move_64.h5')
        print('モデルmodel_move_64.h5を読み込みました')
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
        dens_t = Dense(64, activation = 'relu',
                       name='dense_T')(teban_input)
        merged = concatenate([dens_1, dens_t])
        dens_2 = Dense(64, activation='relu')(merged)
        move_outputs = []
        for _ in range(64):
            move_outputs.append(Dense(1, activation='sigmoid')
                                 (dens_2))
        model = Model(inputs=[ban_input, teban_input],
                      outputs=move_outputs)
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath = 'model_move_64.h5')
    history = model.fit([b_train, t_train], m_train,
                        epochs=20, batch_size=64,
                        validation_split=0.1, verbose=1,
                        callbacks=[early_stopping, md_ch])
    score = model.evaluate([b_test, t_test], m_test, verbose=0)
    print('Test loss:(model.evaluate)', score[0])
    print('Test accuracy(model.evaluate):', score[1])
    del model