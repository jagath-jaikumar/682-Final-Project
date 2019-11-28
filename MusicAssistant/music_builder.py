import pickle
import numpy as np
from GeneratorNet import predict as GenNet
from ClassifierNet import predict as ClassNet

sequence_length = 100


def initialize_all():
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    network_input_flat, network_input_shaped = GenNet.prepare_input_sequence(all_notes)
    # Create 2 networks, 1 dark 1 light so we don't have to keep reloading the weights
    light_generator = GenNet.create_network(network_input_shaped, "light", len(all_notes))
    dark_generator = GenNet.create_network(network_input_shaped, "dark", len(all_notes))
    classifier = ClassNet.create_network()
    return light_generator, dark_generator, classifier, network_input_shaped


def run_models(light_generator, dark_generator, classifier, network_input, song_length):
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    full_output = []
    curr_input = network_input
    moods = []
    query_interval = 20# , 50
    j = 1
    i = 0
    while i < song_length:
        class_input = np.reshape(curr_input, (1, sequence_length))
        print("classifier")
        print(class_input.shape)
        if classifier.predict(class_input) == "dark":
            print("predicted dark")
            while j % query_interval != 0:
                note_name, pattern = GenNet.generate_next_note(dark_generator, curr_input, int_to_ps, ps_to_note_name)
                full_output.append(note_name)
                curr_input = list(curr_input.flatten())
                curr_input.append(pattern)
                curr_input = curr_input[-100:]
                curr_input = np.reshape(curr_input, (1, 100, 1))
                j += 1
                i += 1
        else:
            print("predicted light")
            while j % query_interval != 0:
                print(type(curr_input), j)
                note_name, pattern = GenNet.generate_next_note(light_generator, curr_input, int_to_ps, ps_to_note_name)
                full_output.append(note_name)
                curr_input = list(curr_input.flatten())
                curr_input.append(pattern)
                curr_input = curr_input[-100:]
                curr_input = np.reshape(curr_input, (1, 100, 1))
                j += 1
                i += 1
        print("done with iteration ", i+1)
    return full_output, moods


if __name__ == '__main__':
    for i in range(5):
        light_generator, dark_generator, classifier, network_input = initialize_all()
        song_length = 100
        generated_song, moods = run_models(light_generator, dark_generator, classifier, network_input, song_length)
        filename = "GeneratedSongs/"
        filename += 'Mixed' + str(i)

        GenNet.create_midi(generated_song, filename)
