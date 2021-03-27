# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

DEMoS dataset organization 

DOWNLOAD THE .ZIP FILE FROM HERE:

https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi%3A10.5683%2FSP2%2FE8H2MF#

AND PLACE IT AT: data/raw/

"""

# %% Imports
import sys 
sys.path.append('..')
import os
import zipfile
import shutil

# %% Unzip
path_raw = '../../data/raw/'
with zipfile.ZipFile(path_raw + 'dataverse_files.zip', 'r') as zip_ref:
    path_extract = path_raw + 'TESS_audio_speech'
    if not os.path.exists(path_extract):
        os.mkdir(path_extract)
    zip_ref.extractall(path_extract)


# %% Categorizar dados em pastas
path = '../../data/raw/TESS_audio_speech/'
path_out = '../../data/raw/TESS_Emotions'

for subdir, dirs, files in os.walk(path):
    for file in files: 
        if file.find("wav") != -1: # only get the wav files
            if file.find("neutral") != -1:
                path_paste = path_out + '/0 - Neutral/'
            elif file.find("happy") != -1:
                path_paste = path_out + '/1 - Happy/'
            elif file.find("sad") != -1: 
                path_paste = path_out + '/2 - Sad/'
            elif file.find("angry") != -1:
                path_paste = path_out + '/3 - Anger/'
            elif file.find("fear") != -1: 
                path_paste = path_out + '/4 - Fear/'    
            elif file.find("disgust") != -1:
                path_paste = path_out + '/5 - Disgust/'
            elif file.find("ps") != -1:
                path_paste = path_out + '/6 - Surprise/'
        else:
            continue
            
        # Criar caminho caso não exista
        if not os.path.exists(path_paste):
            os.makedirs(path_paste)
        # Colar arquivos
        os.replace(path + file, path_paste + file)

# Delete empty folders
shutil.rmtree(path)  