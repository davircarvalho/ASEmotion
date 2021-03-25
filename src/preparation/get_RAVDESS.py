# imports 1
import os
import requests
import sys 
sys.path.append('..')

folders = os.listdir('../../data/')

if 'raw' in folders:
    if os.path.isfile('../../data/raw/RAVDESS_Audio_Speech.zip'):
        print('You already have a file called RAVDESS_Audio_Speech.zip')
    else:
        data_url = 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
        data = requests.get(data_url)
        with open('../../data/raw/RAVDESS_Audio_Speech.zip','wb') as f:
            f.write(data.content)
        print('Dataset downloaded successfully')
else:
    mydir = os.getcwd()
    print('You are running from a non expected directory')
    print(f'Current directory: {mydir}')