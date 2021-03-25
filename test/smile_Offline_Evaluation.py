# -*- coding: utf-8 -*-
'''
 Test AEmotion offline performance
'''
# %% Import
from tensorflow.keras.models import model_from_json

import numpy as np
import os
import sys 
sys.path.append('..')
from tcn import TCN, tcn_full_summary
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import opensmile

import tensorflow as tf
import tensorflow_io as tfio


# %% Resample
def load_wav(filename):
    """ read in a waveform file and convert to 44.1 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          contents=file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=44100)
    
    # Trim silence
    # position = tfio.experimental.audio.trim(wav, axis=0, epsilon=0.001)
    # processed = wav[position[0] : position[1]]
    return wav.numpy(), 44100



# %% Load files 
path = '../dataset/Italiano/4 - Anger'

# Config opensmile feature set
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

lst = []
i = -2
for subdir, dirs, files in os.walk(path):
    i+=1
    print(subdir)
    print(i)
    for file in files:
        # Load file
        filename = os.path.join(subdir,file)
        data, Fs = load_wav(filename)
        
        # Make chunks 
        N = int(np.floor(3*Fs))  # Number of samples
        k_chunks = int(np.floor(np.size(data)/N)) # Number of chunks available in one audio
        chunk_data = np.empty(shape=(k_chunks, N))
        if k_chunks >= 1:
            for k in range(0, k_chunks):
                data_chunk = data[k*N : k*N+N]
            
                # Opensmile
                X_smile = smile.process_signal(data_chunk, Fs)
        
                # Append to list 
                arr = X_smile.values, i
                lst.append(arr)
        
print(np.shape(lst))

# Array conversion
x_test, y = zip(*lst)
x_test = np.array(x_test)


#%% Input normalization
def scale_dataset(x_in):
    scaler = MinMaxScaler()
    y_out = np.empty(shape=(np.shape(x_in)))
    for k in range(np.shape(x_in)[0]):        
        y_out[k,:,:] = scaler.fit_transform(x_in[k,:,:])
    return y_out

x = scale_dataset(x_test)
# x = x.astype('float32')


# %% load saved model 
loaded_json = open('../Network/model_smile_it.json', 'r').read()
model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

tcn_full_summary(model, expand_residual_blocks=False)

# restore weights
model.load_weights('../Network/weights_smile_it.h5')


# %% Prediction
pred = model.predict(x)
predi = pred.argmax(axis=1)
c=0
labels = ['Guilt', 'Disgust', 'Happy', 'Fear', 'Anger', 'Surprise', 'Sad']
for i, n in enumerate(predi):
    if n == 4:
        c += 1  
    print("Audio " + str(i) + ": " + labels[n])
print(c/i*100, "%")