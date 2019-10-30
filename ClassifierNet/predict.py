import glob
import pickle
import numpy
import pandas as pd
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


sequence_length = 100


def generate(filename):
    network_input = get_notes(filename)
    model = create_network(network_input)
    label = prediction(model, network_input)

def get_notes(filename):

    midi = converter.parse(filename)
    notes = []
    

    network_input = []

    print("Parsing %s" % filename)

    notes_to_parse = None

    try:
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    for i in range(0, len(notes)-sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length))
    network_input = network_input / float(len(notes))

    return network_input[0], pitchnames

def create_network(network_input):
    """ create the structure of the neural network """
    print(len(network_input[0]))
    model = Sequential()
    model.add(Dense(128, activation='relu',))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    model.load_weights('classification-improvement-10.hdf5')

    return model


def prediction(model, network_input):
    label = model.predict(network_input)
    print(label)


if __name__ == '__main__':
    filename = "midi_songs/8.mid"
    generate(filename)