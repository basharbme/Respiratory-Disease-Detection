import audio_processing as ap
import os
import re
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

def extract_info(filename):
    splits = filename.split('_')
    print(splits)

def main():
    # separate files by audio and txt
    wav_files, txt_files = get_audio_files()
    wav_files = sorted(wav_files)

    # extract information from the wav files' filename
    # 0:[patient #], 1:[recording index], 2:[chest location], 3:[acquisition mode], 4:[recording equipment]
    stripped_file_info = [re.sub('\.wav$', '', file) for file in wav_files]
    file_info = [file.split('_') for file in stripped_file_info]
    
    # load the extracted data into a dataframe
    demog_data = pd.read_csv('demographic-info.csv')
    diag_data = pd.read_csv('respiratory-sound-database/patient_diagnosis.csv', names=['Patient Number', 'Diagnosis'])
    data = pd.DataFrame(data=file_info, columns=['Patient Number', 'Recording Index', 'Chest Location', 'Acquisition Mode', 'Recording Equipment'])
    data['Patient Number'] = data['Patient Number'].astype(int)

    # put info from demog_data into data
    age, sex, a_BMI, c_weight, c_height, diagnosis = [], [], [], [], [], []
    demog_size = demog_data['Age'].size
    size = data['Patient Number'].size

    # TODO: Make this more efficient
    for j in range(0, size):
        for i in range(0, demog_size):
            if data['Patient Number'][j] == demog_data['Patient Number'][i]:
                age.append(demog_data['Age'][i])
                sex.append(demog_data['Sex'][i])
                a_BMI.append(demog_data['Adult BMI'][i])
                c_weight.append(demog_data['Child Weight'][i])
                c_height.append(demog_data['Child Height'][i])
            if data['Patient Number'][j] == diag_data['Patient Number'][i]:
                diagnosis.append(diag_data['Diagnosis'][i])
            pass
    
    data['Age'], data['Sex'], data['Adult BMI'], data['Child Weight'], data['Child Height'], data['Diagnosis'] = age, sex, a_BMI, c_weight, c_height, diagnosis

    # get list of unique diagnoses labels
    diagnoses = data.Diagnosis.unique()

    print(sorted(txt_files))
    # spectros = ap.get_spectrograms(wav_files)




if __name__ == "__main__":
    main()