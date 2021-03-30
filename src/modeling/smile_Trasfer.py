'''
# -*- coding: utf-8 -*-"""
Created on Mon Mar  1 16:48:20 2021

@author: rdavi
'''

# %% Import libs
import sys 
sys.path.append('..')

from src.modeling.tcn.tcn import TCN, compiled_tcn

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
import seaborn as sb
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Activation, Dense

# %% Define the INPUT dataset
# dataset = 'DEMOS'
# dataset = 'RAVDESS'
dataset = 'TESS'
# dataset = 'RAVDESS_TESS'
# dataset = 'AEMOTION'


# %% Load dataset 
with open('../../data/processed/dataset_smile_' +dataset+ '.pckl', 'rb') as f:
    [X, y] = pickle.load(f)
    
    
# %% Filter inputs and targets
# Split between train and test 
x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=y)

# # Input normalization
def scale_dataset(x_in):
    scaler = MinMaxScaler()
    y_out = np.empty(shape=(np.shape(x_in)))
    for k in range(np.shape(x_in)[2]):        
        y_out[:,:,k] = scaler.fit_transform(x_in[:,:,k])
    return y_out

x_train = scale_dataset(x_train)
x_test = scale_dataset(x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# %% Define the trained model 
# dataset = 'DEMOS'
dataset = 'RAVDESS'
# dataset = 'TESS'
# dataset = 'AEMOTION'

# Load model from file
with open('../../model/model_smile_' +dataset+ '.json', 'r') as json_file:
    loaded_json = json_file.read()
    base_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
    # restore weights
    base_model.load_weights('../../model/weights_smile_' +dataset+ '.h5')


# %% Freeze model and add new layers
base_model.trainable = False  # Freeze the outer model
base_model.summary()

inputs = tf.keras.Input(shape=(x_train[0:1].shape[1], x_train.shape[2]))
# We make sure that the base_model is running in inference mode here
x = base_model(inputs, training=False)
x = Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
x = Activation('relu')(x)
outputs = Dense(7)(x)
x = Activation('softmax')(outputs)
model = Model(inputs, outputs)

model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')


# %% Train
early_stop = EarlyStopping(monitor="val_accuracy", patience=8)
cnnhistory = model.fit(x_train, y_train,
                        batch_size = 32,
                        validation_data=(x_test, y_test),
                        epochs = 100,
                        verbose = 1,
                        callbacks=early_stop)



# %% Save it all
# get model as json string and save to file
model_as_json = model.to_json()
with open('../../model/model_transfer_' +dataset+ '.json', 'w') as json_file:
    json_file.write(model_as_json)
    # save weights to file (for this format, need h5py installed)
    model.save_weights('../../model/weights_transfer_' +dataset+ '.h5')



# %% Plot accuracy n loss
# h = plt.figure()
# plt.plot(cnnhistory.history['loss'])
# plt.plot(cnnhistory.history['val_loss'])
# plt.title('Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.grid()
# plt.show()
# h.savefig('../../data/Loss' +dataset+ '.pdf', bbox_inches='tight')


# h = plt.figure()
# plt.plot(cnnhistory.history['accuracy'])
# plt.plot(cnnhistory.history['val_accuracy'])
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Test'], loc='lower right')
# plt.grid()
# plt.show()
# h.savefig('../../data/Accuracy' +dataset+ '.pdf', bbox_inches='tight')


# # %% Confusion Matrix
# lb = LabelEncoder()
# pred = model.predict(x_test, verbose=1)
# pred = pred.squeeze().argmax(axis=1)
# new_y_test = y_test.astype(int)

# mtx = confusion_matrix(new_y_test, pred)
# labels = ['Neutral', 'Happy', 'Sad', 'Anger', 'Fear', 'Disgust', 'Surprise']
# h = plt.figure()
# sb.heatmap(mtx, annot = True, fmt ='d',
#            yticklabels=labels,
#            xticklabels=labels,
#            cbar=False)
# plt.title('Confusion matrix')
# h.savefig('../../data/Confusion' +dataset+ '.pdf', bbox_inches='tight')
# %%
