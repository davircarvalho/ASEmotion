"""
Real time AEmotion
"""
# %% Import libs
import pyaudio
import numpy as np
import pickle
import librosa
import keract

import sys 
sys.path.append('..')
from tcn import TCN

from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime as dtime
import opensmile


import paho.mqtt.client as mqtt #import the client1
import time


# %% Initialize MQTT
def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    # print("message qos=",message.qos)
    # print("message retain flag=",message.retain)


def on_log(client, userdata, level, buf):
    print("log: ",buf)


broker_address="146.164.26.62"
broker_port = 2494
keepalive = 60

print("creating new instance")
client = mqtt.Client("Labinter02") #create new instance
client.on_message=on_message #attach function to callback
# client.on_log=on_log
client.username_pw_set("participants", "prp1nterac")
print("connecting to broker")
client.connect(broker_address, broker_port, keepalive)

client.loop_start() #start the loop


# %% reload saved model 
# load model from file
with open('../Network/model_smile_it.json', 'r') as json_file:
    loaded_json = json_file.read()
    model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
    # restore weights
    model.load_weights('../Network/weights_smile_it.h5')


# %% Pre-process input
# Config for opensmile feature set
smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


def input_prep(data, smile):
    X_smile = np.empty(shape=(1, 296, 25))
    df_x = smile.process_signal(data, 44100)
    scaler = MinMaxScaler()
    X_smile[0,:,:] = scaler.fit_transform(df_x.values)
    return X_smile


# %% Identificar dispositivos de audio do sistema
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


#  Time streaming #############################################
RATE = 44100 # Sample rate
nn_time = 3 # signal length send to the network
CHUNK = round(RATE*nn_time) # Frame size

print('janela de análise é de: {0} segundos'.format(CHUNK/RATE))
#input stream setup
# pyaudio.paInt16 : representa resolução em 16bit 
stream=p.open(format = pyaudio.paFloat32,
                       rate=RATE,
                       channels=1, 
                       input_device_index = 5,
                       input=True,  
                       frames_per_buffer=CHUNK)


labels = ['Guilt', 'Disgust', 'Happy', 'Neutral', 'Anger', 'Surprise', 'Sad']
history_pred = []
hist_time = []
while True:
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    x_infer = input_prep(data, smile)
    pred = model.predict(x_infer)
    predi = pred.argmax(axis=1)
    history_pred = np.append(history_pred, predi[0])
    # hist_time = np.append(hist_time, dtime.now().strftime('%H:%M:%S'))
    print(labels[predi[0]] + "  --  (raw data peak: " + str(max(data))+")")
    
    # GET ACTIVATIONS
    layername = 'activation_3' 
    l_weights = keract.get_activations(model, x_infer, layer_names=layername)
    w_values = np.squeeze(l_weights[layername])
    
    # SEND TO MQTT BrOKER
    client.publish('hiper/labinter99_ita', labels[predi[0]])
    for k in range(len(labels)):
        topic_pub = "hiper/labinter_ita_"+labels[k]
        # client.subscribe(topic_pub)
        client.publish(topic_pub, str(w_values[k]))
        
        # SEND TO MQTT BrOKER
        # for k in range(len(labels)):
        #     mqtt_client.publish_single(float(w_values[k]), topic=labels[k])

        # plot
        # clear_output(wait=True)
        # plt.plot(w_values, 'b-')
        # plt.title(labels[predi[0]])
        # plt.yticks(ticks=np.arange(0,1.1,0.1))
        # plt.xticks(ticks=np.arange(0,7), labels=labels)
        # plt.xlabel('Emotion')
        # plt.ylabel('NN certainty')
        # plt.grid()
        # plt.show()  


# %% Plot history 

h=plt.figure()
plt.scatter(range(0,len(history_pred)), history_pred)
plt.yticks(range(0,7) , labels=labels)
# plt.xticks(range(0,len(history_pred)) , labels=hist_time, rotation=90)


plt.xlabel('Time (each dot represents a ' + str(nn_time)+ 's iteration)')
plt.ylabel('Emotion')
plt.title('AEmotion classification')
plt.grid()
plt.show()
h.savefig("hist.pdf", bbox_inches='tight')