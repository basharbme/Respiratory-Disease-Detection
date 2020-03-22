import audio_processing as ap
import os
import pandas as pd
import numpy as np

def get_audio_files():
    path = 'respiratory-sound-database/audio_and_txt_files/'
    folder = os.listdir(path)

    wav_files, txt_files = [], []
    for file in folder:
        if file.endswith('.wav'):
            wav_files.append(file)
        elif file.endswith('.txt'):
            txt_files.append(file)

    return wav_files, txt_files

def main():
    wav_files, txt_files = get_audio_files()
    spectros = ap.get_spectrograms(wav_files)
    df = pd.read_csv('demographic-info.csv')
    print(df.head())


if __name__ == "__main__":
    main()