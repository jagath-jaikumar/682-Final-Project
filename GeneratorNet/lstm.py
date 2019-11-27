import glob
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

use_cuda = True
if use_cuda:
    from tensorflow.keras.layers import CuDNNLSTM

save_all = True


def hyperparameter_tuning(feeling):
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    n_vocab = len(all_notes)
    print(n_vocab)

    if feeling == 'dark':
        with open('../Data/lstm_dark_data.pkl', 'rb') as f:
            training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)
        print(training_inputs.shape, training_outputs.shape)
        filepath = "weights/Dark-LSTM-improvement-{epoch:02d}.hdf5"
    else:
        with open('../Data/lstm_light_data.pkl', 'rb') as f:
            training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)
        print(training_inputs.shape, training_outputs.shape)

        filepath = "weights/Light-LSTM-improvement-{epoch:02d}.hdf5"
    if save_all:
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            mode='min'
        )
    else:
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
    model = Sequential()
    if use_cuda:
        model.add(CuDNNLSTM(512, input_shape=(training_inputs.shape[1], training_inputs.shape[2]),
                        return_sequences=True))
        model.add(CuDNNLSTM(512, return_sequences=True, ))
        model.add(CuDNNLSTM(512))
    else:
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
    model.add(Dense(n_vocab)) # Should be 81 because there are 81 notes the output can be
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    for layer in model.layers:
        print(layer.output_shape)
    if save_all:
        if feeling == 'dark':
            history = model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
                      validation_split = 0.25,epochs=150, batch_size=128, shuffle=True, verbose=1, callbacks=[checkpoint])
        else:
            history = model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
                      validation_split = 0.25,epochs=250, batch_size=128, shuffle=True, verbose=1, callbacks=[checkpoint])

    # print(history.history)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('ModelLoss={}.png'.format(feeling))
    plt.show()


if __name__ == '__main__':
    hyperparameter_tuning(feeling="light")


# Create Light LSTM
# def light_hyperparameter_tuning():
#     with open('lstm_light.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#         training_inputs, training_outputs, validation_inputs, validation_outputs = pickle.load(f)
#
#     filepath = "Light-LSTM-improvement-{epoch:02d}.hdf5"
#     checkpoint = ModelCheckpoint(
#         filepath,
#         monitor='loss',
#         verbose=0,
#         save_best_only=True,
#         mode='min'
#     )
#     # nothing above .001, GOOD LR = .00001
#     model = Sequential()
#     model.add(LSTM(
#         512,
#         input_shape=(network_input.shape[1], network_input.shape[2]),
#         return_sequences=True
#     ))
#     model.add(Dropout(0.3))
#     model.add(LSTM(512, return_sequences=True))
#     model.add(Dropout(0.3))
#     model.add(LSTM(512))
#     model.add(Dense(256))
#     model.add(Dropout(0.3))
#     model.add(Dense(n_vocab))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     for layer in model.layers:
#         print(layer.output_shape)
#     model.fit(training_inputs, training_outputs, validation_data=(validation_inputs, validation_outputs),
#               epochs=300, batch_size=bs, shuffle=True, verbose=2)  # , callbacks=[checkpoint])
