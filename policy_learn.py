import os.path
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def readkifu(kifu_list_file):
    print(f'{kifu_list_file}を読み込んでいます')
    banlist = np.empty((0,128))
    banlists = np.empty((0,128))
    labelist = np.empty((0, 64))
    labelists = np.empty((0, 64))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            k = np.load(kifu)
            ban = k[:,:-1]
            move = k[:,-1]
            label = np.eye(64)[move]
            banlist = np.append(banlist, ban, axis=0)
            labelist = np.append(labelist, label, axis=0)
            cnt += 1
            if cnt % 100 == 0:
                banlists = np.append(banlists, banlist, axis=0)
                labelists = np.append(labelists, labelist, axis=0)
                banlist = np.empty((0,128))
                labelist = np.empty((0, 64))

        print(f'{kifu_list_file}を読み込みました')
    return banlists, labelists

if __name__ == '__main__':
    banlist_train, labelist_train = readkifu('kifulist_train.txt')
    banlist_test, labelist_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_policy.h5'):
        model = load_model('model_policy.h5')
        print('モデルmodel_policy.h5を読み込みました')
    else:
        print('モデルはありません')
        model = Sequential()
        model.add(Dense(500, input_dim=128, activation='relu'))
        model.add(Dense(500,activation='relu'))
        model.add(Dense(500,activation='relu'))
        model.add(Dense(500,activation='relu'))
        model.add(Dense(500,activation='relu'))
        model.add(Dense(500,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    md_ch = ModelCheckpoint(filepath = 'model_policy.h5')
    history = model.fit(banlist_train, labelist_train,
                        epochs=20, batch_size=32,
                        validation_split=0.1, callbacks=[early_stopping, md_ch])
    with open("history.pickle", mode='wb') as f:
        pickle.dump(history.history, f)

    score = model.evaluate(banlist_test, labelist_test, verbose=0)
    pred = model.predict(banlist_test[-50:,],batch_size=1)
    classes = model.predict_classes(banlist_test[-50:,],batch_size=1)
    print('Test loss:(model.evaluate)', score[0])
    print('Test accuracy(model.evaluate):', score[1])
    print()
    print('predict:(model.predict)')
    print(pred)
    print()
    print('classes:(model.predict_classes)')
    print(classes)
    true_classes = np.argmax(labelist_test[-50:,], axis=1)
    print('ture_classes:')
    print(true_classes)
    model.save('model_policy.h5')
    del model
