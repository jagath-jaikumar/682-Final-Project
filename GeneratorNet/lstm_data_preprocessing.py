import glob
import pickle
import numpy as np
import pandas as pd
import random
import sys
from music21 import converter, instrument, note, chord


sequence_length = 100


def map_light_songs_to_notes():
    data = pd.read_csv("song_category.csv")
    lstm_train_in = []
    lstm_train_out = []
    lstm_val_in = []
    lstm_val_out = []
    with open('notes.pkl', 'rb') as f:
        note_mapping = pickle.load(f)
    train_indices = np.random.choice(53, 41, replace=False)
    index = 0
    for file in glob.glob("../Data/light/*.mid"):
        song_name = str(file).split("/")[-1]
        notes_for_song = []
        midi = converter.parse(file)
        print("Parsing files for map_songs_to_notes %s" % file)
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes_for_song.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes_for_song.append('.'.join(str(n) for n in element.normalOrder))
        for i in range(len(notes_for_song) - sequence_length):
            sequence_in = notes_for_song[i:i + sequence_length]
            next_note = notes_for_song[i + sequence_length]
            if index in train_indices:
                lstm_train_in.append([note_mapping[char] for char in sequence_in])
                lstm_train_out.append(note_mapping[next_note])
            else:
                lstm_val_in.append([note_mapping[char] for char in sequence_in])
                lstm_val_out.append(note_mapping[next_note])
        index += 1

    with open('lstm_light.pkl', 'wb') as f:
        pickle.dump([lstm_train_in, lstm_train_out, lstm_val_in, lstm_val_out], f)


def map_dark_songs_to_notes():
    data = pd.read_csv("song_category.csv")
    lstm_train_in = []
    lstm_train_out = []
    lstm_val_in = []
    lstm_val_out = []
    with open('notes.pkl', 'rb') as f:
        note_mapping = pickle.load(f)
    train_indices = np.random.choice(39, 31, replace=False)
    index = 0
    for file in glob.glob("../Data/dark/*.mid"):
        song_name = str(file).split("/")[-1]
        notes_for_song = []
        midi = converter.parse(file)
        print("Parsing files for map_songs_to_notes %s" % file)
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes_for_song.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes_for_song.append('.'.join(str(n) for n in element.normalOrder))
        for i in range(len(notes_for_song) - sequence_length):
            sequence_in = notes_for_song[i:i + sequence_length]
            next_note = notes_for_song[i + sequence_length]
            if index in train_indices:
                lstm_train_in.append([note_mapping[char] for char in sequence_in])
                lstm_train_out.append(note_mapping[next_note])
            else:
                lstm_val_in.append([note_mapping[char] for char in sequence_in])
                lstm_val_out.append(note_mapping[next_note])
        index += 1

    with open('lstm_dark.pkl', 'wb') as f:
        pickle.dump([lstm_train_in, lstm_train_out, lstm_val_in, lstm_val_out], f)


def map_inputs_to_output():
    with open('lstm_light.pkl', 'rb') as f:
        lstm_train_in_l, lstm_train_out_l, lstm_val_in_l, lstm_val_out_l = pickle.load(f)
    with open('lstm_dark.pkl', 'rb') as f:
        lstm_train_in_d, lstm_train_out_d, lstm_val_in_d, lstm_val_out_d = pickle.load(f)
    # LIGHT zip, shuffle, resave
    c = list(zip(lstm_train_in_l, lstm_train_out_l))
    random.shuffle(c)
    lstm_train_in_l, lstm_train_out_l = zip(*c)
    c = list(zip(lstm_val_in_l, lstm_val_out_l))
    random.shuffle(c)
    lstm_val_in_l, lstm_val_out_l = zip(*c)

    #  DARK zip, shuffle, resave
    c = list(zip(lstm_train_in_d, lstm_train_out_d))
    random.shuffle(c)
    lstm_train_in_d, lstm_train_out_d = zip(*c)
    c = list(zip(lstm_val_in_d, lstm_val_out_d))
    random.shuffle(c)
    lstm_val_in_d, lstm_val_out_d = zip(*c)

    # LIGHT
    training_inputs_l = np.asarray(lstm_train_in_l)
    training_outputs_l = np.asarray(lstm_train_out_l)
    validation_inputs_l = np.asarray(lstm_val_in_l)
    validation_outputs_l = np.asarray(lstm_val_out_l)
    n_patterns = len(training_inputs_l)
    training_inputs_l = np.reshape(training_inputs_l, (n_patterns, sequence_length))
    n_patterns = len(lstm_val_in_l)
    validation_inputs_l = np.reshape(lstm_val_in_l, (n_patterns, sequence_length))

    # DARK
    training_inputs_d = np.asarray(lstm_train_in_d)
    training_outputs_d = np.asarray(lstm_train_out_d)
    validation_inputs_d = np.asarray(lstm_val_in_d)
    validation_outputs_d = np.asarray(lstm_val_out_d)
    n_patterns = len(training_inputs_d)
    training_inputs_d = np.reshape(training_inputs_d, (n_patterns, sequence_length))
    n_patterns = len(lstm_val_in_d)
    validation_inputs_d = np.reshape(lstm_val_in_d, (n_patterns, sequence_length))

    with open('lstm_light.pkl', 'wb') as f:
        pickle.dump([training_inputs_l, training_outputs_l, validation_inputs_l, validation_outputs_l], f)
    with open('lstm_dark.pkl', 'wb') as f:
        pickle.dump([training_inputs_d, training_outputs_d, validation_inputs_d, validation_outputs_d], f)


if __name__ == '__main__':
    # map_dark_songs_to_notes()
    map_inputs_to_output()