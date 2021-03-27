# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

Unite RAVDESS and TESS datasets: Make sure you've run "Organize_RAVDESS.py"
and "Organize_TESS.py" befoore running this script
"""


# %% Imports
import sys 
sys.path.append('..')
import os
from distutils.dir_util import copy_tree

# Criar caminho caso não exista
path_out = '../../data/raw/RAVDESS_TESS_Emotions/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

# %% Copy folders together
copy_tree('../../data/raw/TESS_Emotions/', path_out)
copy_tree('../../data/raw/RAVDESS_Emotions/', path_out)

# %%
