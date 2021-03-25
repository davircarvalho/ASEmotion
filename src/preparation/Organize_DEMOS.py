# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

DEMoS dataset organization 

THIS DATASET IS NOT PUBLICLY AVAILABLE, BUT CAN BE REQUESTED AT:
https://zenodo.org/record/2544829/accessrequest
"""

# %% Imports
import sys 
sys.path.append('..')
import os

# %% Categorizar dados em pastas
path = '../../data/raw/DEMoS/DEMOS/'
path_out = '../../data/raw/Italiano'

for subdir, dirs, files in os.walk(path):
    for file in files:
        emotion = file[8:11] # if DEMoS
        if emotion == 'col': # guilt
            path_paste = path_out + '/0 - Guilt/'
        elif emotion == 'dis': # disgust 
            path_paste = path_out + '/1 - Disgust/'
        elif emotion == 'gio': # happy 
            path_paste = path_out + '/2 - Happy/'
        elif emotion == 'pau' or emotion == 'ans': # fear 
            path_paste = path_out + '/3 - Fear/'
        elif emotion == 'rab': # anger
            path_paste = path_out + '/4 - Anger/'
        elif emotion == 'sor': # surprise
            path_paste = path_out + '/5 - Surprise/'
        elif emotion == 'tri': # sadness
            path_paste = path_out + '/6 - Sad/'
        elif emotion == 'neu': # neutral
            path_paste = path_out + '/7 - Neutral/'
            
        # Criar caminho caso não exista
        if not os.path.exists(path_paste):
            os.makedirs(path_paste)
        # Colar arquivos
        os.replace(path + file, path_paste + file)
        