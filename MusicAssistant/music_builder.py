import pickle
import GeneratorNet
import ClassifierNet


def initialize_all():
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    network_input = GeneratorNet.prepare_input_sequence(all_notes)
    # Create 2 networks, 1 dark 1 light so we don't have to keep reloading the weights
    light_generator = GeneratorNet.create_network(network_input, len(all_notes), "light")
    dark_generator = GeneratorNet.create_network(network_input, len(all_notes), "dark")
    classifier = ClassifierNet.create_network()
    return light_generator, dark_generator, classifier, network_input


def run_models(light_generator, dark_generator, classifier, network_input, song_length):
    with open('../Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    full_output = []
    curr_input = network_input
    for i in range(song_length):
        if classifier.predict(curr_input) == "dark":
            next_note, new_input = GeneratorNet.generate_next_note(dark_generator, network_input, int_to_ps,
                                                                   ps_to_note_name)
            full_output.append(next_note)
        else:
            next_note, new_input = GeneratorNet.generate_next_note(light_generator, network_input, int_to_ps,
                                                                   ps_to_note_name)
            full_output.append(next_note)
        curr_input = new_input

    return full_output


if __name__ == '__main__':
    light_generator, dark_generator, classifier, network_input = initialize_all()
    song_length = 10
    generated_song = run_models(light_generator, dark_generator, classifier, network_input, song_length)
    GeneratorNet.create_midi(generated_song)
