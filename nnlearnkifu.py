import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense
from keras.models import Sequential

def readkifu(kifu):
    k = np.load(kifu)
    b = k[0:-1]
    l = k[-1]
    if l[0] == 1:
        label = np.array([[0, 1] for i in range(b.shape[0])])
    elif l[0] == 0:
        label = np.array([[1, 0] for i in range(b.shape[0])])
    return b, label

if __name__ == '__main__':
    banlist, labelist = readkifu('./kifu/18_07_07_11_00_10-1480.npy')
    model = Sequential()
    model.add(Dense(200, input_dim=128, activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(banlist, labelist, epochs=10, batch_size=32)
    classes = model.predict_classes(banlist, batch_size=1)
    prob = model.predict_proba(banlist, batch_size=1)
    print('classfied:')
    print(labelist == classes)
    print()
    print('output probability:')
    print(prob)
