import glob
import pickle
import numpy as np
import pandas as pd
import random
import sys
from music21 import converter, instrument, note, chord


sequence_length = 100


def get_notes():
    notes = []
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
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        pitchnames = sorted(set(item for item in notes))
        # mapping of chords/notes to values - this is what the sequences need to use
        note_to_int = dict((note1, number) for number, note1 in enumerate(pitchnames))
        with open('notes.pkl', 'wb') as f:
            pickle.dump(note_to_int, f)
    # return notes


def map_songs_to_notes():
    data = pd.read_csv("song_category.csv")
    song_feeling = data.set_index('song_title').T.to_dict('list')
    network_input = []
    network_output = []
    light_dark_song_dict = {'light': [], 'dark': []}
    song_to_shingles_dict = {}
    num_notes = 0
    with open('notes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        note_mapping = pickle.load(f)
    for file in glob.glob("midi_songs/*.mid"):
        song_name = str(file).split("/")[-1]
        song_title = song_name.split('\\')[1]
        category = song_feeling[song_title][0]
        notes_for_song = []
        light_dark_song_dict[category].append(song_title)
        midi = converter.parse(file)
        song_name = str(file).split("/")[-1]
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
        song_to_shingles_dict[song_title] = []
        for i in range(len(notes_for_song) - sequence_length + 1):
            sequence_in = notes_for_song[i:i + sequence_length]
            temp = [note_mapping[char] for char in sequence_in]
            song_to_shingles_dict[song_title].append(temp)
    light_shingles_training = []
    dark_shingles_training = []
    light_shingles_val = []
    dark_shingles_val = []
    light_song_indices = np.random.choice(53, 31, replace=False)
    dark_song_indices = np.random.choice(39, 31, replace=False)
    print("light and dark song indices", light_song_indices, dark_song_indices)
    print(np.arange(53), np.arange(53).shape)
    light_val_indices = np.setdiff1d(np.arange(53), light_song_indices).tolist()
    dark_val_indices = np.setdiff1d(np.arange(39), dark_song_indices).tolist()
    light_val_indices = np.random.choice(np.arange(len(light_val_indices)), len(dark_val_indices), replace=False)
    print("validation light indices: ", light_val_indices)
    print("validation dark indices: ", dark_val_indices)

    # training
    light_song_title_training = []
    dark_song_title_training = []
    for index in light_song_indices:
        light_song_title_training.append(light_dark_song_dict["light"][index])
    print(light_song_title_training)
    for index in dark_song_indices:
        dark_song_title_training.append(light_dark_song_dict["dark"][index])
    print(dark_song_title_training)
    for song_title in light_song_title_training:
        light_shingles_training.extend(song_to_shingles_dict[song_title])
    print("total light shingles", len(light_shingles_training))
    for song_title in dark_song_title_training:
        dark_shingles_training.extend(song_to_shingles_dict[song_title])
    print("total dark shingles", len(dark_shingles_training))

    # validation
    light_song_title_validation = []
    dark_song_title_validation = []
    for index in light_val_indices:
        light_song_title_validation.append(light_dark_song_dict["light"][index])
    print(light_song_title_validation)
    for index in dark_val_indices:
        dark_song_title_validation.append(light_dark_song_dict["dark"][index])
    print(dark_song_title_validation)
    for song_title in light_song_title_validation:
        light_shingles_val.extend(song_to_shingles_dict[song_title])
    print("total light shingles", len(light_shingles_val))
    for song_title in dark_song_title_validation:
        dark_shingles_val.extend(song_to_shingles_dict[song_title])
    print("total dark shingles", len(dark_shingles_val))

    with open('training_data_raw.pkl', 'wb') as f:
        pickle.dump([light_song_title_training, dark_song_title_training, light_song_indices, dark_song_indices,
                     light_shingles_training, dark_shingles_training], f)
    with open('validation_data_raw.pkl', 'wb') as f:
        pickle.dump([light_song_title_validation, dark_song_title_validation, light_val_indices, dark_val_indices,
                     light_shingles_val, dark_shingles_val], f)

    print("training label lengths: Light: ", len(light_shingles_training), "Dark: ", len(dark_shingles_training))
    print("validation label lengths: Light: ", len(light_shingles_val), "Dark: ", len(dark_shingles_val))
    data_counts = {"light_train": len(light_shingles_training), "dark_train": len(dark_shingles_val),
                   "light_val": len(light_shingles_val)}
    with open('data_info.pkl', 'wb') as f:
        pickle.dump(data_counts, f)


def map_inputs_to_output():
    with open('training_data_raw.pkl', 'rb') as f:
        light_song_title_training, dark_song_title_training, light_song_indices, dark_song_indices, \
            light_shingles_training, dark_shingles_training = pickle.load(f)
    light_labels_raw = np.array([0, 1])
    dark_labels_raw = np.array([1, 0])
    random.shuffle(light_shingles_training)
    random.shuffle(dark_shingles_training)
    num_samples = min(len(light_shingles_training), len(dark_shingles_training))
    light_labels = np.asarray([light_labels_raw, ] * num_samples)
    dark_labels = np.asarray([dark_labels_raw, ] * num_samples)
    training_shingles = np.concatenate((light_shingles_training[0:num_samples],
                                        dark_shingles_training[0:num_samples]), axis=0)
    training_labels = np.concatenate((light_labels, dark_labels), axis=0)
    c = list(zip(training_shingles, training_labels))
    random.shuffle(c)
    training_shingles, training_labels = zip(*c)

    with open('validation_data_raw.pkl', 'rb') as f:
        light_song_title_validation, dark_song_title_validation, light_val_indices, dark_val_indices, \
            light_shingles_val, dark_shingles_val = pickle.load(f)
    random.shuffle(light_shingles_val)
    random.shuffle(dark_shingles_val)
    num_samples = min(len(light_shingles_val), len(dark_shingles_val))
    light_labels = np.asarray([light_labels_raw,] * num_samples)
    dark_labels = np.asarray([dark_labels_raw,] * num_samples)

    validation_shingles = np.concatenate((light_shingles_val[0:num_samples],
                                          dark_shingles_val[0:num_samples]), axis=0)
    validation_labels = np.concatenate((light_labels, dark_labels), axis=0)
    c = list(zip(validation_shingles, validation_labels))
    random.shuffle(c)
    validation_shingles, validation_labels = zip(*c)

    training_shingles = np.asarray(training_shingles)
    training_labels = np.asarray(training_labels)
    validation_shingles = np.asarray(validation_shingles)
    validation_labels = np.asarray(validation_labels)
    n_patterns = len(training_shingles)
    training_shingles = np.reshape(training_shingles, (n_patterns, sequence_length))
    n_patterns = len(validation_shingles)
    validation_shingles = np.reshape(validation_shingles, (n_patterns, sequence_length))
    print(training_shingles.shape, training_labels.shape)
    print(validation_shingles.shape, validation_labels.shape)
    with open('final_training_data.pkl', 'wb') as f:
        pickle.dump([training_shingles, training_labels], f)
    with open('final_validation_data.pkl', 'wb') as f:
        pickle.dump([validation_shingles, validation_labels], f)


if __name__ == '__main__':
    map_songs_to_notes()
    map_inputs_to_output()
