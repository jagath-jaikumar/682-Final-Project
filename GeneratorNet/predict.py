import pickle
import numpy
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
    with open('../Data/notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    notes = notes[0]

    network_input_flat,network_input_shaped = prepare_sequences(notes)

    model = create_network(network_input_shaped, feeling)

    prediction_output = generate_notes(model,network_input_flat, notes)

    create_midi(prediction_output)

def prepare_sequences(notes):
    network_input_flat = []

    for i in range(sequence_length):
        sequence_in = notes[numpy.random.randint(0, len(notes)-1)]
        network_input_flat.append(sequence_in)

    network_input_shaped = numpy.reshape(network_input_flat,(1,sequence_length,1))
    return network_input_flat,network_input_shaped

def create_network(training_inputs,feeling):
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
    model.add(Dense(81)) # should be n_vocab but is wrong dimension?
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if feeling == 'dark':
        model.load_weights('weights/Dark-LSTM-improvement-150.hdf5')
    else:
        model.load_weights('light-LSTM-improvement-150.hdf5')
    return model

def generate_notes(model,network_input_flat,notes):

    prediction_output = []

    pattern = network_input_flat

    for new_note in range(5):

        pattern = numpy.reshape(pattern,(1,sequence_length,1))
        new_prediction = model.predict(pattern, verbose = 0)
        index = numpy.argmax(new_prediction)
        result = notes[index]
        print(result)
        prediction_output.append(result)

        pattern = list(pattern.flatten())
        pattern.append(result)
        pattern = pattern[1:]

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for n in prediction_output:
        new_note = note.Note(n)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # for pattern in prediction_output:
    #     if ('.' in pattern) or pattern.isdigit():
    #         notes_in_chord = pattern.split('.')
    #         notes = []
    #         for current_note in notes_in_chord:
    #             new_note = note.Note(int(current_note))
    #             new_note.storedInstrument = instrument.Piano()
    #             notes.append(new_note)
    #         new_chord = chord.Chord(notes)
    #         new_chord.offset = offset
    #         output_notes.append(new_chord)
    #     else:
    #         new_note = note.Note(pattern)
    #         new_note.offset = offset
    #         new_note.storedInstrument = instrument.Piano()
    #         output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate(feeling='dark')
