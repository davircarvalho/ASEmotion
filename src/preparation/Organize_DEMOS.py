# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

DEMoS dataset organization 

THIS DATASET IS NOT PUBLICLY AVAILABLE, BUT CAN BE REQUESTED AT:
https://zenodo.org/record/2544829/accessrequest

After downloading it, place the .zip file at: /data/raw/
"""

# %% Imports
import sys 
sys.path.append('..')
import os
import zipfile
import shutil


# %% unzip
path_raw = '../../data/raw/'
with zipfile.ZipFile(path_raw + 'wav_DEMoS.zip', 'r') as zip_ref:
    path_extract = path_raw + 'DEMoS'
    if not os.path.exists(path_extract):
        os.mkdir(path_extract)
    zip_ref.extractall(path_extract)


# %% Copy to desired structure
path = '../../data/raw/DEMoS/DEMOS/'
path_out = '../../data/raw/DEMOS_Emotions'
os.replace('../../data/raw/DEMoS/NEU/', path_out + '/0 - Neutral/')

for subdir, dirs, files in os.walk(path):
    for file in files:
        emotion = file[8:11] # if DEMoS
        if emotion == 'neu': # neutral
            path_paste = path_out + '/0 - Neutral/'
        elif emotion == 'gio': # happy 
            path_paste = path_out + '/1 - Happy/'
        elif emotion == 'tri': # sadness
            path_paste = path_out + '/2 - Sad/'
        elif emotion == 'rab': # anger
            path_paste = path_out + '/3 - Anger/'
        elif emotion == 'pau' or emotion == 'ans': # fear 
            path_paste = path_out + '/4 - Fear/'
        elif emotion == 'dis': # disgust 
            path_paste = path_out + '/5 - Disgust/'
        elif emotion == 'sor': # surprise
            path_paste = path_out + '/6 - Surprise/'
        elif emotion == 'col': # guilt
            path_paste = path_out + '/7 - Guilt/'            
        # Criar caminho caso não exista
        if not os.path.exists(path_paste):
            os.makedirs(path_paste)
        # Colar arquivos
        os.replace(path + file, path_paste + file)

# Delete empty folders
shutil.rmtree(path_extract)
        
# %%
