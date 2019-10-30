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

def train_network():
    network_input, network_output = get_notes()
    model = create_network(network_input)
    train(model, network_input, network_output)



def get_notes():
    data = pd.read_csv("song_category.csv")

    dic = data.set_index('song_title').T.to_dict('list')

    notes = []
    
    network_input = []
    network_output = []

    num_notes = 0

    for file in glob.glob("midi_songs/training_subset/*.mid"):
        midi = converter.parse(file)
        song_name = str(file).split("/")[-1]

        print("Parsing %s" % file)

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


        category = dic[song_name][0]



        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            if category == 'dark':
                network_output.append(0)
            else:
                network_output.append(1)
            network_input.append([note_to_int[char] for char in sequence_in])
        num_notes += len(notes)

    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length))

    network_input = network_input / float(num_notes)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)



def create_network(network_input):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')


    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "classification-improvement-{epoch:02d}.hdf5"
    bs = 16
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=10, batch_size=bs, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
