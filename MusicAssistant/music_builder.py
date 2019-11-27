import pickle
from GeneratorNet import predict as GenNet
from ClassifierNet import predict as ClassNet


def initialize_all():
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    network_input_flat, network_input_shaped = GenNet.prepare_input_sequence(all_notes)
    # Create 2 networks, 1 dark 1 light so we don't have to keep reloading the weights
    light_generator = GenNet.create_network(network_input_shaped, len(all_notes), "light")
    dark_generator = GenNet.create_network(network_input_shaped, len(all_notes), "dark")
    classifier = ClassNet.create_network()
    return light_generator, dark_generator, classifier, network_input_shaped


def run_models(light_generator, dark_generator, classifier, network_input, song_length):
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    full_output = []
    curr_input = network_input
    for i in range(song_length):
        print(curr_input.shape)
        class_input = curr_input.reshape(curr_input.shape[1:])
        print(class_input.shape)
        if classifier.predict(class_input) == "dark":
            next_note, new_input = GenNet.generate_next_note(dark_generator, network_input, int_to_ps,
                                                                   ps_to_note_name)
            print(x_train.shape[1])
            sys.exit()
            full_output.append(next_note)
        else:
            next_note, new_input = GenNet.generate_next_note(light_generator, network_input, int_to_ps,
                                                                   ps_to_note_name)
            full_output.append(next_note)
        print("done with iteration ", i)
        curr_input = new_input

    return full_output


if __name__ == '__main__':
    light_generator, dark_generator, classifier, network_input = initialize_all()
    song_length = 10
    generated_song = run_models(light_generator, dark_generator, classifier, network_input, song_length)
    GenNet.create_midi(generated_song)
