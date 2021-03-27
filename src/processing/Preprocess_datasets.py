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




# %% Define the dataset
path = "D:/Documentos/1 - Work/AEmotion/dataset/emotion_portuguese_database"


# %% Resample
def load_wav(filename):
     wav, fs = librosa.load(filename, sr = 16000)
     return wav, fs

# Initialize opensmile feature set
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)   
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

        # Make chunks 
        N = int(np.floor(duration*Fs))  # Number of samples in two second
        k_chunks = int(np.floor(np.size(data)/N)) # Number of chunks available in one audio
        data_chunk = np.empty(shape=(N))
        if k_chunks >= 1:
            for k in range(0, k_chunks):
                data_chunk = data[k*N : k*N+N]
            
                # Opensmile
                X_smile = smile.process_signal(data_chunk, Fs)
        
                # Append to list 
                arr = X_smile.values, i
                lst.append(arr)
        else: # zero pad at the end if audio is less than specified duration 
            data_chunk[:np.size(data)] = data
            
            # Opensmile
            X_smile = smile.process_signal(data_chunk, Fs)
    
            # Append to list 
            arr = X_smile.values, i
            lst.append(arr)


# %% Save smile dataset
X, y = zip(*lst)
X, y = np.asarray(X), np.asarray(y)
with open('Network/dataset_smile_ptbr_16khz.pckl', 'wb') as f:
    pickle.dump([X, y], f)
print("All done!")
# %%