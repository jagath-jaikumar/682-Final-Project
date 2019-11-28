import pickle
import numpy
from GeneratorNet import predict as GenNet
from ClassifierNet import predict as ClassNet

sequence_length = 100

def initialize_all():
    with open('Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    network_input_flat, network_input_shaped = GenNet.prepare_sequences(all_notes)
    # Create 2 networks, 1 dark 1 light so we don't have to keep reloading the weights
    light_generator = GenNet.create_network(network_input_shaped, "light", len(all_notes))
    dark_generator = GenNet.create_network(network_input_shaped, "dark", len(all_notes))
    classifier = ClassNet.create_network()
    return light_generator, dark_generator, classifier, network_input_shaped, all_notes


def run_models(light_generator, dark_generator, classifier, network_input, song_length, all_notes):
    with open('Data/notes.pkl', 'rb') as f:
        all_notes, ps_to_int, int_to_ps, ps_to_note_name = pickle.load(f)
    full_output = []
    curr_input = network_input
    moods = []
    for i in range(song_length):

        class_input = numpy.reshape(curr_input,(1,sequence_length))
        if classifier.predict(class_input) == "dark":
            next_notes = GenNet.generate_notes(dark_generator,curr_input,all_notes, phrase_length = 100)
            moods.append('dark')
        else:
            next_notes = GenNet.generate_notes(light_generator,curr_input,all_notes, phrase_length = 100)
            moods.append('light')

        for note in next_notes:
            full_output.append(note)

        print("done with iteration ", i+1)
        curr_input = full_output[-100:]

        # this is my change

    return full_output, moods


if __name__ == '__main__':
    for i in range(5):
        light_generator, dark_generator, classifier, network_input, all_notes = initialize_all()
        song_length = 5
        generated_song, moods = run_models(light_generator, dark_generator, classifier, network_input, song_length, all_notes)
        dcount, lcount = 0,0
        for j in moods:
            if j == 'dark':
                dcount+=1
            else:
                lcount+=1
        filename = "GeneratedSongs/"
        if dcount > lcount:
            filename+="Dark" + str(i+1)
        else:
            filename+='Light'+ str(i+1)

        GenNet.create_midi(generated_song, filename)
