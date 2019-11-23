import pickle
import numpy as np
import random
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

sequence_length = 100


def hyperparameter_tuning():
    with open('../Data/classification_train_val_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        x_train, y_train, x_val, y_val = pickle.load(f)
    print(x_train.shape, y_train.shape)
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
    for l_rate in [.0001]:
        # [100, 150, 200]
        for bs in [125]:
            model = Sequential()
            model.add(Dense(100, input_shape=(x_train.shape[1],), kernel_initializer='random_normal', activation='relu'))
            model.add(Dense(60, kernel_initializer='random_normal', activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(60, kernel_initializer='random_normal', activation='relu'))
            # model.add(BatchNorm())
            model.add(Dense(45, kernel_initializer='random_normal', activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
            model.add(Dense(2, kernel_initializer='random_normal', activation='softmax'))
            adam = optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=bs, shuffle=True,
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
