import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


# Create Light LSTM
def light_hyperparameter_tuning():
    with open('lstm_light.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)

    filepath = "Light-LSTM-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    # nothing above .001, GOOD LR = .00001
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
              epochs=300, batch_size=bs, shuffle=True, verbose=2)  # , callbacks=[checkpoint])
    # y_pred = model.predict(x_val)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_real = np.argmax(y_val, axis=1)
    # cm = confusion_matrix(y_real, y_pred)
    # print(cm)
    # precision = cm[0][0] / (np.sum(cm[0]))
    # recall = cm[0][0] / (np.sum(cm[0][0] + cm[1][0]))
    # f_measure = (2*recall*precision)/(recall+precision)
    # print("Precision = ", precision)
    # print("Recall = ", recall)
    # print("f-measure = ", f_measure)


# Create Dark LSTM
def dark_hyperparameter_tuning():
    with open('lstm_light.pkl', 'rb') as f:
        training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)

    filepath = "Light-LSTM-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    # nothing above .001, GOOD LR = .00001
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
              epochs=300, batch_size=bs, shuffle=True, verbose=2)  # , callbacks=[checkpoint])
    # y_pred = model.predict(x_val)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_real = np.argmax(y_val, axis=1)
    # cm = confusion_matrix(y_real, y_pred)
    # print(cm)
    # precision = cm[0][0] / (np.sum(cm[0]))
    # recall = cm[0][0] / (np.sum(cm[0][0] + cm[1][0]))
    # f_measure = (2*recall*precision)/(recall+precision)
    # print("Precision = ", precision)
    # print("Recall = ", recall)
    # print("f-measure = ", f_measure)


if __name__ == '__main__':
    pass
