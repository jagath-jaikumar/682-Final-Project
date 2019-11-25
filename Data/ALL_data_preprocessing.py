import glob
import pickle
import numpy as np
import pandas as pd
import random
import sys
import tensorflow.keras.utils as utils
from music21 import converter, instrument, note, chord


sequence_length = 100


def get_notes():
    pitches = []
    ps_to_note_name = {}
    # get the notes from all the midi files and enumerate them
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("Parsing Files for get_notes %s" % file)
        notes_to_parse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitches.append(element.pitch.ps)
                ps_to_note_name[element.pitch.ps] = str(element.pitch)
            elif isinstance(element, chord.Chord):
                # chord.pitches
                for pitch in element.pitches:
                    pitches.append(pitch.ps)
                    ps_to_note_name[pitch.ps] = str(pitch)
    all_possible_note_frequencies = list(set(pitches))
    all_possible_note_frequencies.sort()
    ps_to_int = dict((ps, number) for number, ps in enumerate(all_possible_note_frequencies))
    int_to_ps = dict((number, ps) for number, ps in enumerate(all_possible_note_frequencies))
    with open('notes.pkl', 'wb') as f:
        pickle.dump([all_possible_note_frequencies, ps_to_int, int_to_ps, ps_to_note_name], f)


def map_songs_to_notes():
    dark_train_in = []
    dark_lstm_train_out = []
    dark_class_train_out = []
    dark_val_in = []
    dark_lstm_val_out = []
    dark_class_val_out = []
    with open('notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)

    train_indices = np.random.choice(39, 31, replace=False)
    index = 0
    for file in glob.glob("dark/*.mid"):
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
                notes_for_song.append(element.pitch.ps)
            elif isinstance(element, chord.Chord):
                for pitch in element.pitches:
                    notes_for_song.append(pitch.ps)
        for i in range(len(notes_for_song) - sequence_length):
            sequence_in = notes_for_song[i:i + sequence_length]
            next_note = notes_for_song[i + sequence_length]
            if index in train_indices:
                dark_train_in.append(sequence_in)
                dark_class_train_out.append(0)
                dark_lstm_train_out.append(ps_to_int[next_note])
            else:
                dark_val_in.append(sequence_in)
                dark_class_val_out.append(0)
                dark_lstm_val_out.append(ps_to_int[next_note])
        index += 1
    dark_lstm_train_out = utils.to_categorical(dark_lstm_train_out, num_classes=81)
    dark_lstm_val_out = utils.to_categorical(dark_lstm_val_out, num_classes=81)
    dark_class_train_out = utils.to_categorical(dark_class_train_out, num_classes=2)
    dark_class_val_out = utils.to_categorical(dark_class_val_out, num_classes=2)

    a = list(zip(dark_train_in, dark_lstm_train_out))
    b = list(zip(dark_train_in, dark_class_train_out))
    c = list(zip(dark_val_in, dark_lstm_val_out))
    d = list(zip(dark_val_in, dark_class_val_out))
    random.shuffle(a)
    random.shuffle(b)
    random.shuffle(c)
    random.shuffle(d)
    dark_train_in_lstm, dark_lstm_train_out = zip(*a)
    dark_train_in_class, dark_class_train_out = zip(*b)
    dark_val_in_lstm, dark_lstm_val_out = zip(*c)
    dark_val_in_class, dark_class_val_out = zip(*d)

    light_train_in = []
    light_lstm_train_out = []
    light_class_train_out = []
    light_val_in = []
    light_lstm_val_out = []
    light_class_val_out = []
    train_indices = np.random.choice(53, 42, replace=False)
    index = 0
    for file in glob.glob("light/*.mid"):
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
                notes_for_song.append(element.pitch.ps)
            elif isinstance(element, chord.Chord):
                for pitch in element.pitches:
                    notes_for_song.append(pitch.ps)
        for i in range(len(notes_for_song) - sequence_length):
            sequence_in = notes_for_song[i:i + sequence_length]
            next_note = notes_for_song[i + sequence_length]
            if index in train_indices:
                light_train_in.append(sequence_in)
                light_class_train_out.append(1)
                light_lstm_train_out.append(ps_to_int[next_note])
            else:
                light_val_in.append(sequence_in)
                light_class_val_out.append(1)
                light_lstm_val_out.append(ps_to_int[next_note])
        index += 1
    light_lstm_train_out = utils.to_categorical(light_lstm_train_out, num_classes=81)
    light_lstm_val_out = utils.to_categorical(light_lstm_val_out, num_classes=81)
    light_class_train_out = utils.to_categorical(light_class_train_out, num_classes=2)
    light_class_val_out = utils.to_categorical(light_class_val_out, num_classes=2)
    a = list(zip(light_train_in, light_lstm_train_out))
    b = list(zip(light_train_in, light_class_train_out))
    c = list(zip(light_val_in, light_lstm_val_out))
    d = list(zip(light_val_in, light_class_val_out))
    random.shuffle(a)
    random.shuffle(b)
    random.shuffle(c)
    random.shuffle(d)
    light_train_in_lstm, light_lstm_train_out = zip(*a)
    light_train_in_class, light_class_train_out = zip(*b)
    light_val_in_lstm, light_lstm_val_out = zip(*c)
    light_val_in_class, light_class_val_out = zip(*d)

    ############ LSTM DATA CONVERSION ############################
    # SAVE LSTM DATA BECAUSE THIS IS ALL WE NEED FOR LSTM
    training_inputs_l = np.asarray(light_train_in_lstm) / float(1)
    training_outputs_l = np.asarray(light_lstm_train_out)
    validation_inputs_l = np.asarray(light_val_in_lstm) / float(1)
    validation_outputs_l = np.asarray(light_lstm_val_out)
    n_patterns = len(training_inputs_l)
    training_inputs_l = np.reshape(training_inputs_l, (n_patterns, sequence_length, 1))
    n_patterns = len(validation_inputs_l)
    validation_inputs_l = np.reshape(validation_inputs_l, (n_patterns, sequence_length, 1))

    # DARK
    training_inputs_d = np.asarray(dark_train_in_lstm) / float(1)
    training_outputs_d = np.asarray(dark_lstm_train_out)
    validation_inputs_d = np.asarray(dark_val_in_lstm) / float(1)
    validation_outputs_d = np.asarray(dark_lstm_val_out)
    n_patterns = len(training_inputs_d)
    training_inputs_d = np.reshape(training_inputs_d, (n_patterns, sequence_length, 1))
    n_patterns = len(validation_inputs_d)
    validation_inputs_d = np.reshape(validation_inputs_d, (n_patterns, sequence_length, 1))

    print("Light LSTM")
    print(training_inputs_l.shape, training_outputs_l.shape)
    print(validation_inputs_l.shape, validation_outputs_l.shape)
    print("Dark LSTM")
    print(training_inputs_d.shape, training_outputs_d.shape)
    print(validation_inputs_d.shape, validation_outputs_d.shape)

    with open('lstm_light_data.pkl', 'wb') as f:
        pickle.dump([training_inputs_l, training_outputs_l, validation_inputs_l, validation_outputs_l], f)
    with open('lstm_dark_data.pkl', 'wb') as f:
        pickle.dump([training_inputs_d, training_outputs_d, validation_inputs_d, validation_outputs_d], f)

    ############ MORE STEPS FOR CLASSIFICATION
    # CONTINUE TO DO MORE STUFF FOR CLASSIFICATION
    light_train_in_class = np.asarray(light_train_in_class)
    dark_train_in_class = np.asarray(dark_train_in_class)
    light_class_train_out = np.asarray(light_class_train_out)
    dark_class_train_out = np.asarray(dark_class_train_out)
    num_samples = min(len(light_train_in_class), len(dark_train_in_class))
    training_class_in = np.concatenate((light_train_in_class[0:num_samples],
                                        dark_train_in_class[0:num_samples]), axis=0)
    training_class_out = np.concatenate((light_class_train_out[0:num_samples],
                                        dark_class_train_out[0:num_samples]), axis=0)
    light_val_in_class = np.asarray(light_val_in_class)
    dark_val_in_class = np.asarray(dark_val_in_class)
    light_class_val_out = np.asarray(light_class_val_out)
    dark_class_val_out = np.asarray(dark_class_val_out)
    num_samples = min(len(light_val_in_class), len(dark_val_in_class))
    val_class_in = np.concatenate((light_val_in_class[0:num_samples],
                                   dark_val_in_class[0:num_samples]), axis=0)
    val_class_out = np.concatenate((light_class_val_out[0:num_samples],
                                    dark_class_val_out[0:num_samples]), axis=0)
    a = list(zip(training_class_in, training_class_out))
    b = list(zip(val_class_in, val_class_out))
    random.shuffle(a)
    random.shuffle(b)
    training_class_in, training_class_out = zip(*a)
    val_class_in, val_class_out = zip(*b)

    # Postprocess data to create numpy arrays and save to pickle files
    training_shingles = np.asarray(training_class_in)
    training_labels = np.asarray(training_class_out)
    validation_shingles = np.asarray(val_class_in)
    validation_labels = np.asarray(val_class_out)
    n_patterns = len(training_shingles)
    training_shingles = np.reshape(training_shingles, (n_patterns, sequence_length))
    n_patterns = len(validation_shingles)
    validation_shingles = np.reshape(validation_shingles, (n_patterns, sequence_length))
    print("Classification")
    print(training_shingles.shape, training_labels.shape)
    print(validation_shingles.shape, validation_labels.shape)
    with open('classification_train_val_data.pkl', 'wb') as f:
        pickle.dump([training_shingles, training_labels, validation_shingles, validation_labels], f)


if __name__ == '__main__':
    get_notes()
    map_songs_to_notes()
