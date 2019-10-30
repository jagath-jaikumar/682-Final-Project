import pandas as pd
import glob
import os
from shutil import copyfile


if __name__ == "__main__":
    data = pd.read_csv("../Data/song_category.csv")

    dic = data.set_index('song_title').T.to_dict('list')

    for file in glob.glob("../Data/midi_songs/*.mid"):
        song_name = str(file).split("/")[-1]
        category = dic[song_name][0]


        if category == 'light':
            new_filename = "../Data/light/" + song_name + ".mid"
            copyfile(str(file), new_filename)
        else:
            new_filename = "../Data/dark/" + song_name + ".mid"
            copyfile(str(file), new_filename)
