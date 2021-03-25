'''
# -*- coding: utf-8 -*-"""
Created on Mon Mar  1 16:48:20 2021

@author: rdavi
'''

# %% Import libs
import sys 
sys.path.append('..')

from tcn import TCN, compiled_tcn

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
# import seaborn as sb
from tensorflow.keras.callbacks import EarlyStopping



# %% Load dataset 
with open('../Network/dataset_smile_rav_tess_16khz.pckl', 'rb') as f:
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
    for k in range(np.shape(x_in)[0]):        
        y_out[k,:,:] = scaler.fit_transform(x_in[k,:,:])
    return y_out

x_train = scale_dataset(x_train)
x_test = scale_dataset(x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# %% Create TCN
model = compiled_tcn(return_sequences=False,
                    num_feat=x_train.shape[2],
                    num_classes=len(np.unique(y_train)),
                    nb_filters=65,
                    kernel_size=9,
                    dilations=[2 ** i for i in range(7)], 
                    nb_stacks=1,
                    dropout_rate=0.2,
                    use_weight_norm=True,
                    max_len=x_train[0:1].shape[1],
                    use_skip_connections=True,
                    opt='adam',
                    lr=0.002226)
model.summary()


# %% Train
early_stop = EarlyStopping(monitor="val_accuracy", patience=4)

cnnhistory = model.fit(x_train, y_train,
                        batch_size = 38,
                        validation_data=(x_test, y_test),
                        epochs = 70,
                        verbose = 1,
                        callbacks=early_stop)



# %% Save it all
# get model as json string and save to file
model_as_json = model.to_json()
with open('../Network/model_smile_rav_tess_16khz.json', 'w') as json_file:
    json_file.write(model_as_json)
    # save weights to file (for this format, need h5py installed)
    model.save_weights('../Network/weights_smile_rav_tess_16khz.h5')





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
# h.savefig("../Network/Loss.pdf", bbox_inches='tight')


# h = plt.figure()
# plt.plot(cnnhistory.history['accuracy'])
# plt.plot(cnnhistory.history['val_accuracy'])
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Test'], loc='lower right')
# plt.grid()
# plt.show()
# h.savefig("../Network/Accuracy.pdf", bbox_inches='tight')




# # %% reload saved model 
# # load model from file
# with open('../Network/model_smile_it.json', 'r') as json_file:
#     loaded_json = json_file.read()
#     model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
#     # restore weights
#     model.load_weights('../Network/weights_smile_it.h5')



# # %% Confusion Matrix
# lb = LabelEncoder()
# pred = model.predict(x_test, verbose=1)
# pred = pred.squeeze().argmax(axis=1)
# new_y_test = y_test.astype(int)

# mtx = confusion_matrix(new_y_test, pred)
# labels = ['Guilt', 'Disgust', 'Happy', 'Fear', 'Anger', 'Surprise', 'Sad']
# h = plt.figure()
# sb.heatmap(mtx, annot = True, fmt ='d',
#            yticklabels=labels,
#            xticklabels=labels,
#            cbar=False)
# plt.title('Confusion matrix')
# h.savefig("../Network/Confusion.pdf", bbox_inches='tight')