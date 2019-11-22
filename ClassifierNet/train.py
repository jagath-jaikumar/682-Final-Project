import pickle
import numpy as np
import random
import sys
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

sequence_length = 100


def hyperparameter_tuning():
    with open('final_training_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        x_train, y_train = pickle.load(f)
    with open('final_validation_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        x_val, y_val = pickle.load(f)
    filepath = "classification-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    # nothing above .001, GOOD LR = .00001
    # [.00001, .0001, .001]
    for l_rate in [.001]:
        # [100, 150, 200]
        for bs in [150]:
            model = Sequential()
            model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
            # model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
            model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu'))
            # model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu'))
            model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu'))
            # model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
            model.add(Dense(2, kernel_initializer='glorot_normal', activation='softmax'))
            adam = optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batch_size=bs, shuffle=True,
                      verbose=2)#, callbacks=[checkpoint])
            y_pred = model.predict(x_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_real = np.argmax(y_val, axis=1)
            cm = confusion_matrix(y_real, y_pred)
            print(cm)
            precision = cm[0][0] / (np.sum(cm[0]))
            recall = cm[0][0] / (np.sum(cm[0][0] + cm[1][0]))
            f_measure = (2*recall*precision)/(recall+precision)
            print("Precision = ", precision)
            print("Recall = ", recall)
            print("f-measure = ", f_measure)


if __name__ == '__main__':
    hyperparameter_tuning()
