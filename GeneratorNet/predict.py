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
    print(notes)

    n_vocab = len(set(notes))

    print(pitchnames)
    print(n_vocab)

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab, feeling)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    print(prediction_output)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    network_input = []

    for i in range(sequence_length):
        sequence_in = notes[numpy.random.randint(0, len(sequence_in)-1)]
        network_input.append(sequence_in)

    return network_input

def create_network(training_inputs, n_vocab,feeling):
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
    model.add(Dense(353)) # should be n_vocab but is wrong dimension?
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if feeling == 'dark':
        model.load_weights('dark-LSTM-improvement-150.hdf5')
    else:
        model.load_weights('light-LSTM-improvement-150.hdf5')
    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    start = numpy.random.randint(0, len(network_input)-1)
    print("pitchnamnes")
    print(pitchnames)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))


    print('int to note')
    print(int_to_note)
    pattern = network_input[start]
    prediction_output = []
    prev_prediction = 0
    for note_index in range(5):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)


        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction[0])

        print(prediction[0][index])

        result = int_to_note[index]

        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        prev_prediction = prediction

    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate(feeling='dark')
