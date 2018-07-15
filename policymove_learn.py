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
    banlist = np.empty((0,2,8,8))
    tebanlist = np.empty((0, 1))
    movelist = np.empty((0, 64))
    resultlist = np.empty((0,2))
    banlists = np.empty((0,2,8,8))
    tebanlists = np.empty((0, 1))
    movelists = np.empty((0, 64))
    resultlists = np.empty((0,2))
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
                move = i[129:193].reshape((1,64))
                result = np.eye(2)[i[193]].reshape((1,2))
                banlist = np.append(banlist, ban, axis=0)
                tebanlist = np.append(tebanlist, teban, axis=0)
                movelist = np.append(movelist, move, axis=0)
                resultlist = np.append(resultlist, result, axis=0)
            cnt += 1
            if cnt % 20 ==0:
                banlists = np.append(banlists, banlist, axis=0)
                tebanlists = np.append(tebanlists, tebanlist, axis=0)
                movelists = np.append(movelists, movelist, axis=0)
                resultlists = np.append(resultlists, resultlist, axis=0)
                banlist = np.empty((0,2,8,8))
                tebanlist = np.empty((0, 1))
                movelist = np.empty((0, 64))
                resultlist = np.empty((0,2))
        print(f'{kifu_list_file}を読み込みました')
    movelists = movelists.astype('int32')
    resultlists = resultlists.astype('int32')
    return banlists, tebanlists, movelists, resultlists

if __name__ == '__main__':
    b_train, t_train, m_train, r_train = readkifu('kifulist_train.txt')
    b_test, t_test, m_test, r_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_policy_value.h5'):
        model = load_model('model_policy_value.h5')
        print('モデルmodel_policy_value.h5を読み込みました')
    else:
        print('モデルはありません')
        ban_input = Input(shape=(2, 8, 8))
        conv_1 = Conv2D(192, (3,3), padding='same')(ban_input)
        b = BatchNormalization(axis=1)(conv_1)
        r = Activation('relu')(b)
        conv_2 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_2)
        r = Activation('relu')(b)
        conv_3 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_3)
        r = Activation('relu')(b)
        conv_4 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_4)
        r = Activation('relu')(b)
        conv_5 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_5)
        r = Activation('relu')(b)
        conv_6 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_6)
        r = Activation('relu')(b)
        conv_7 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_7)
        r = Activation('relu')(b)
        conv_8 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_8)
        r = Activation('relu')(b)
        conv_9 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_9)
        r = Activation('relu')(b)
        conv_10 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_10)
        r = Activation('relu')(b)
        conv_11 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_11)
        r = Activation('relu')(b)
        conv_12 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_12)
        r = Activation('relu')(b)
        conv_13 = Conv2D(192, (3,3), padding='same')(r)
        b = BatchNormalization(axis=1)(conv_13)
        r = Activation('relu')(b)
        x = Flatten()(r)
        dens_1 = Dense(256, activation = 'relu')(x)
        teban_input = Input(shape=(1,))
        dens_t = Dense(256,activation='relu',name='dense_T')(teban_input)
        merged = concatenate([dens_1, dens_t])
        dens_2 = Dense(512, activation='relu')(merged)
        dens_3 = Dense(256, activation='relu')(dens_2)
        dens_4 = Dense(128, activation='relu')(dens_3)
        move_output = Dense(64, activation='softmax')(dens_4)
        result_output = Dense(2, activation='softmax')(dens_4)
        model = Model(inputs=[ban_input, teban_input], outputs=[move_output, result_output])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  loss_weights=[0.5, 0.5],
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath = 'model_policy_value.h5')
    history = model.fit([b_train, t_train], [m_train, r_train],
                        epochs=20, batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stopping, md_ch])
    score = model.evaluate([b_test, t_test], [m_test, r_test], verbose=0)
    pred = model.predict([b_test[-50:], t_test[-50:]], batch_size=1)
    model.save('model_policy_value.h5')
    del model
