## Applied Machine Learning for Affective Computing – Respiratory Disease Detection Project

### Description
Using the dataset found [here](https://www.kaggle.com/vbookshelf/respiratory-sound-database#patient_diagnosis.csv), we compare performances between a Convolutional Neural Network and a Random Forest Classifier to detect the following respiratory diseases from audio of patients' breathing:
- Asthma
- COPD (Chronic Obstructive Pulmonary Disorder)
- LRTI (Lower Respiratory Tract Infection)
- URTI (Upper Respiratory Tract Infection)
- Bronchiectasis
- Bronchiolitis
- Pneumonia

### Notes
The script `changebit.sh` included in this repository is meant to change the bit depth of the .wav files found in the dataset (for use with scipy's `spectrogram` module). The files from the dataset originally came as 24-bit files, which are not supported by `spectrogram`. The script overwrites the files at the existing directory.

### Features
We convert the .wav files found in the dataset to spectrograms.

### Random Forest Classifier Performance

### Convolutional Neural Network Performance

### Usage
