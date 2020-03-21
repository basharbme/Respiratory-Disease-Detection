import audio_processing
import os

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


if __name__ == "__main__":
    wav_files, txt_files = get_audio_files()

    
