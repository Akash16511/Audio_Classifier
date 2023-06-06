import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
from scipy.io import wavfile as wav
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


#### Extracting MFCC's For every audio file
audio_dataset_path='UrbanSound8K/audio/'
metadata=pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')



def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
def feature_append():
    extracted_features=[]
    for index_num,row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        final_class_labels=row["class"]
        data=features_extractor(file_name)
        extracted_features.append([data,final_class_labels])
        return extracted_features
    

def test(filename,model,labelencoder):
    ### Testing Some Test Audio Data
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    return prediction_class


