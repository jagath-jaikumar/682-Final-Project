import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm

sequence_length = 100
use_cuda = False
if use_cuda:
    from tensorflow.keras.layers import CuDNNLSTM


def generate(feeling, x=100):
    with open('../Data/notes.pkl', 'rb') as f:
        notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)

    network_input_flat, network_input_shaped = prepare_input_sequence(notes)
    model = create_network(network_input_shaped, len(notes), feeling)
    prediction_output = generate_x_notes(model, network_input_flat, int_to_ps, ps_to_note_name, x)
    create_midi(prediction_output)


def prepare_input_sequence(notes):
    network_input_flat = []

    for i in range(sequence_length):
        sequence_in = notes[np.random.randint(0, len(notes)-1)]
        network_input_flat.append(sequence_in)

    network_input_shaped = np.reshape(network_input_flat, (1, sequence_length, 1))
    return network_input_flat, network_input_shaped


def create_network(training_inputs, num_notes, feeling):
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
    model.add(Dense(num_notes)) # should be n_vocab but is wrong dimension?
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if feeling == 'dark':
        model.load_weights('Dark-LSTM-improvement-150.hdf5')
    else:
        model.load_weights('Light-LSTM-improvement-150.hdf5')
    return model


def generate_x_notes(model, network_input_flat, int_to_ps, ps_to_note_name, x):
    prediction_output = []
    pattern = network_input_flat

    for new_note in range(x):
        pattern = np.reshape(pattern, (1, sequence_length, 1))
        new_prediction = model.predict(pattern, verbose=0)
        index = np.argmax(new_prediction)
        ps_value = int_to_ps[index]
        note_name = ps_to_note_name[ps_value]

        print(ps_value, note_name)
        # Adding predicted note to generate the next input
        pattern = list(pattern.flatten())
        pattern.append(ps_value)
        pattern = pattern[1:]

        # Adding note to prediction output
        prediction_output.append(note_name)

    return prediction_output


def generate_next_note(model, network_input_flat, int_to_ps, ps_to_note_name):
    pattern = network_input_flat
    new_prediction = model.predict(pattern, verbose=0)
    index = np.argmax(new_prediction)

    # Adding predicted note to generate the next input
    ps_value = int_to_ps[index]
    pattern.append(ps_value)
    pattern = pattern[1:]

    # Adding note to prediction output
    note_name = ps_to_note_name[ps_value]

    return note_name, pattern


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for n in prediction_output:
        new_note = note.Note(n)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='light4.mid')


if __name__ == '__main__':
    generate(feeling='light', x=100)
