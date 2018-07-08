import os.path
import pickle
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def readkifu(kifu_list_file):
    print(f'{kifu_list_file}を読み込んでいます')
    banlist = np.empty((0,128))
    banlists = np.empty((0,128))
    labelist = np.empty((0, 2))
    labelists = np.empty((0, 2))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
            for line in tqdm(f.readlines()):
                kifu = line.rstrip('\r\n')
                k = np.load(kifu)
                ban = k[0:-1]
                l = k[-1]
                if l[0] == 1:
                    label = np.array([[0, 1] for _ in range(ban.shape[0])])
                elif l[0] == 0:
                    label = np.array([[1, 0] for _ in range(ban.shape[0])])
                banlist = np.append(banlist, ban, axis=0)
                labelist = np.append(labelist, label, axis=0)
                cnt += 1
                if cnt % 500 == 0:
                    banlists = np.append(banlists, banlist, axis=0)
                    labelists = np.append(labelists, labelist, axis=0)
                    banlist = np.empty((0,128))
                    labelist = np.empty((0, 2))

            print(f'{kifu_list_file}を読み込みました')
    return banlists, labelists

if __name__ == '__main__':
    banlist_train, labelist_train = readkifu('kifulist_train.txt')
    banlist_test, labelist_test = readkifu('kifulist_test.txt')
    if os.path.isfile('othlo.h5'):
        model = load_model('othlo.h5')
        print('モデルothlo.h5を読み込みました')
    else:
        print('モデルはありません')
        model = Sequential()
        model.add(Dense(500, input_dim=128, activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    history = model.fit(banlist_train, labelist_train,
                        epochs=100, batch_size=32,
                        validation_split=0.1, callbacks=[early_stopping])
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
    print('confusion_classes:(confusion_matrix(true_classes,model.predict_classes)')
    print(confusion_matrix(true_classes, classes))
    model.save('othlo.h5')
    del model
