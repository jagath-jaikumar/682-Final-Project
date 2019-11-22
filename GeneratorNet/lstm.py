import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import BatchNormalization as BatchNorm
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
    for layer in model.layers:
        print(layer.output_shape)
    model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
              epochs=300, batch_size=bs, shuffle=True, verbose=2)  # , callbacks=[checkpoint])


# Create Dark LSTM
def dark_hyperparameter_tuning():
    with open('notes.pkl', 'rb') as f:
        note_mapping = pickle.load(f)
    n_vocab = len(set(note_mapping))
    print(n_vocab)
    with open('lstm_dark.pkl', 'rb') as f:
        training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)
    print(training_inputs.shape, training_outputs.shape)

    filepath = "Light-LSTM-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    model = Sequential()
    model.add(LSTM(512, input_shape=(training_inputs.shape[1], training_inputs.shape[2]),
                   recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    for layer in model.layers:
        print(layer.output_shape)
    model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
              epochs=150, batch_size=128, shuffle=True, verbose=1)  # , callbacks=[checkpoint])


if __name__ == '__main__':
    dark_hyperparameter_tuning()
