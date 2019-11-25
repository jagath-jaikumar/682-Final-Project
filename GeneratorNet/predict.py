import pickle
import numpy as np
import sys
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm

sequence_length = 100
use_cuda = True
if use_cuda:
    from tensorflow.keras.layers import CuDNNLSTM


def generate(feeling):
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    # print(all_notes)
    # print(ps_to_int)
    # print(int_to_ps)
    # print(ps_to_note_name)

    network_input = prepare_input_sequence(all_notes)
    model = create_network(network_input, len(all_notes), feeling)
    prediction_output = generate_notes(model, network_input, int_to_ps, ps_to_note_name)
    create_midi(prediction_output)


def prepare_input_sequence(all_notes):
    network_input = []

    for i in range(sequence_length):
        sequence_in = all_notes[np.random.randint(0, len(all_notes))]
        network_input.append(sequence_in)

    return network_input


def create_network(training_inputs, n_vocab, feeling):
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
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if feeling == 'dark':
        model.load_weights('dark-LSTM-improvement-150.hdf5')
    else:
        model.load_weights('light-LSTM-improvement-150.hdf5')
    return model


def generate_notes(model, network_input, int_to_ps, ps_to_note_name):
    prediction_output = []
    pattern = network_input

    for new_note in range(5):
        new_prediction = model.predict(pattern, verbose=0)
        index = np.argmax(new_prediction)

        # Adding predicted note to generate the next input
        ps_value = int_to_ps[index]
        pattern.append(ps_value)
        pattern = pattern[1:]

        # Adding note to prediction output
        note_name = ps_to_note_name[ps_value]
        prediction_output.append(note_name)

    return prediction_output


def generate_next_note(model, network_input, int_to_ps, ps_to_note_name):
    pattern = network_input

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

    for pattern in prediction_output:
        # We don't have chords so we only need to go by notes
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    generate(feeling='dark')
