# -*- coding: utf-8 -*-
'''
 Test AEmotion offline performance
'''
# %% Import
from tensorflow.keras.models import model_from_json

import numpy as np
import sys 
sys.path.append('..')
from src.modeling.tcn.tcn import TCN, tcn_full_summary
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sb
import pickle


# %% Define the INPUT dataset
# dataset = 'DEMOS'
# dataset = 'RAVDESS'
# dataset = 'TESS'
dataset = 'AEMOTION'

# LOAD
with open('../data/processed/dataset_smile_' +dataset+ '.pckl', 'rb') as f:
    [X, y_true] = pickle.load(f)
    


# %% Load saved model 
# Define the MODEL dataset
dataset = 'DEMOS'
# dataset = 'RAVDESS'
# dataset = 'TESS'
# dataset = 'RAVDESS_TESS'

# load model from file
with open('../model/model_smile_' +dataset+ '.json', 'r') as json_file:
    loaded_json = json_file.read()
    model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
    # restore weights
    model.load_weights('../model/weights_smile_' +dataset+ '.h5')

tcn_full_summary(model, expand_residual_blocks=False)


#%% Input normalization
def scale_dataset(x_in, fit_scaler=None):
    # Initialize variables
    y_out = np.empty(shape=(np.shape(x_in)))
    save_scaler = []

    # Normalization
    for k in range(np.shape(x_in)[2]): # per feature 
        ### Apply normalization
        if fit_scaler is None: # calculate minmax normalization for training data
            scaler = MinMaxScaler()
            scaler = scaler.fit(x_in[:,:,k])
            save_scaler.append(scaler)
        else: # use the input scaler for the testing data
            scaler = fit_scaler[k]
        y_out[:,:,k] = scaler.transform(x_in[:,:,k])
    return y_out, save_scaler

with open('../model/input_preprocess_' +dataset+ '.pckl', 'rb') as f:
    fit_scaler = pickle.load(f) 

x = scale_dataset(X, fit_scaler)[0]





# %% Prediction
pred = model.predict(x)
predi = pred.argmax(axis=1)

# labels = ['Neutral', 'Happy', 'Sad', 'Anger', 'Fear', 'Disgust', 'Surprise']

# mtx = confusion_matrix(y_true, predi)
# h = plt.figure()
# sb.heatmap(mtx, annot = True, fmt ='d',
#            yticklabels=labels,
#            xticklabels=labels,
#            cbar=False)
# plt.title('Confusion matrix')


# Report
print(classification_report(y_true, predi))

    # %%
