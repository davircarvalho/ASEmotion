    # -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:12:48 2021

@author: rdavi
Preprocess datasets, including silence trimming and spliting in 1s chunks
"""

# %% Import libraries 
import os
import numpy as np
import opensmile
import pickle
import librosa
import matplotlib.pyplot as plt


# %% Define the dataset
dataset = 'DEMOS'
# dataset = 'RAVDESS'
# dataset = 'TESS'

path = '../../data/raw/' +dataset+ '_Emotions/'


# %% Extract features
def load_wav(filename):
    # Load and resample
    audio, fs = librosa.load(filename, sr = 16000)
    # Silence trim 
    interv = librosa.effects.split(audio, top_db=20, frame_length=4096, hop_length=1)
    start, end = interv[0] 
    audio_out = audio[start : end]
    return audio_out, fs

# Initialize opensmile feature set
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors) 

# Sweep  
lst = []
i = -2
duration = 3 # define signal duration in each chunk 
for subdir, dirs, files in os.walk(path):
    i+=1
    print(subdir)
    print(i)
    for file in files:
        # Load file
        filename = os.path.join(subdir,file)
        data, Fs = load_wav(filename)    

        # # Make chunks 
        N = int(np.floor(duration*Fs))  # Number of samples in two second
        data_chunk = np.empty(shape=(N))
        if np.size(data) > N:
            data = data[:N]
        data_chunk[:np.size(data)] = data
        
        # Opensmile
        X_smile = smile.process_signal(data_chunk, Fs)

        # Append to list 
        arr = X_smile.values, i
        lst.append(arr)


# %% Save smile dataset
X, y = zip(*lst)
X, y = np.asarray(X), np.asarray(y)
with open('../../data/processed/dataset_smile_' +dataset+ '.pckl', 'wb') as f:
    pickle.dump([X, y], f)
print("All done!")
# %%
