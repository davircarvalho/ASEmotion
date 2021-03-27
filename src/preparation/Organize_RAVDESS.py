# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

Organize RAVDESS folders according to emotions
"""

# %% Imports
import sys 
sys.path.append('..')
import os
import zipfile
import shutil

# %% Extract zip file
def extract_zip():
    path_raw = '../../data/raw/'
    with zipfile.ZipFile(path_raw + 'RAVDESS_Audio_Speech.zip', 'r') as zip_ref:
        path_extract = path_raw + 'RAVDESS_Audio_Speech'
        if not os.path.exists(path_extract):
            os.mkdir(path_extract)
        zip_ref.extractall(path_extract)
try:
    extract_zip()
except:
    print('RAVDESS_Audio_Speech.zip was not found, downloading it ...')
    import get_RAVDESS
    extract_zip()


# %% Categorizar dados em pastas
path = '../../data/raw/RAVDESS_Audio_Speech/'
path_out = '../../data/raw/RAVDESS_Emotions'

for subdir, dirs, files in os.walk(path):
    for file in files:
        if file[4] == '1': # make sure is speech only
            emotion = int(file[7]) 
            if emotion == 1 or emotion == 2: # neutral / calm
                path_paste = path_out + '/0 - Neutral/'
            elif emotion == 3: # happy 
                path_paste = path_out + '/1 - Happy/'
            elif emotion == 4: # sadness
                path_paste = path_out + '/2 - Sad/'
            elif emotion == 5: # anger
                path_paste = path_out + '/3 - Anger/'
            elif emotion == 6: # fear
                path_paste = path_out + '/4 - Fear/'    
            elif emotion == 7: # disgust    
                path_paste = path_out + '/5 - Disgust/'
            elif emotion == 8: # surprise
                path_paste = path_out + '/6 - Surprise/'
                      
        # Criar caminho caso não exista
        if not os.path.exists(path_paste):
            os.makedirs(path_paste)
        # Colar arquivos
        os.replace(subdir+'/'+file, path_paste + file)
        
# Delete empty folders
shutil.rmtree(path)

# %%
