#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:17:59 2019

@author: poganyg
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = '/media/poganyg/Kaggle/RSNA/rsna-intracranial-hemorrhage-detection/'

df = pd.read_csv(path+'stage_1_train.csv')

# some IDs appear more than once in the dataset
#print(df['ID'].value_counts())

# Checking a few of the duplicate entries it seems that they are duplicates
#df[df['ID'] == "ID_a64d5deed_any"]
"""
ID_489ae4179_subarachnoid
ID_854fba667_intraventricular 
ID_a64d5deed_epidural
ID_a64d5deed_any 
"""

# As they are duplicates they can be dropped from the dataset

#Extracting the ID-s of the duplicate entries
duplicates = df['ID'].value_counts() > 1
print("Number of duplicates: ", sum(duplicates.astype(int)))
duplicateIDs = duplicates.index[duplicates == True]

for duplicateID in duplicateIDs:

    indexes = df.ID[df.ID==duplicateID].index    
    df.drop(indexes[0], inplace=True)

print("Number of duplicates after dropping: ", sum((df['ID'].value_counts() > 1).astype(int)))