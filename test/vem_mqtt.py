# %%
'''
Listen to mqqt broker
'''
import paho.mqtt.client as mqtt #import the client1


# %% Initialize MQTT
def on_message(client, userdata, message):
    
    print("TOPIC: ",message.topic + " --MESSAGE: ",
         (message.payload.decode("utf-8")))
    # print(type(float(message.payload.decode("utf-8"))))
    
    # print("message qos=",message.qos)
    # print("message retain flag=",message.retain)

def on_log(client, userdata, level, buf):
    print("log: ",buf)

def on_connect( client, userdata, flags, rc):
    print ("Connected with Code :" +str(rc))
    # Subscribe Topic from here
    topics = ['hiper/labinter00',
          'hiper/labinter01',
          'hiper/labinter02',
          'hiper/labinter03',
          'hiper/labinter04',
          'hiper/labinter05',
          'hiper/labinter06',
          'hiper/labinter99',]
    for k in range(len(topics)):
        client.subscribe(topics[k])

broker_address="146.164.26.62"
broker_port = 2494
keepalive = 60

print("Creating new client")
client = mqtt.Client("P1") #create new instance
client.on_message=on_message #attach function to callback
# client.on_log=on_log
client.on_connect = on_connect

client.username_pw_set("participants", "prp1nterac")
print("connecting to broker")
client.connect(broker_address, broker_port, keepalive)

# %% 
client.loop_forever() #start the loop

# %%    
