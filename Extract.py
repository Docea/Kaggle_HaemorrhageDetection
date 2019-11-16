import PIL
import pydicom
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from fastai.core import parallel

print("PyTorch version: ", torch.__version__)

'''
    This code is mostly from 'https://github.com/radekosmulski/rsna-intracranial/blob/master/01_start_here.ipynb'
    [slightly adapted]
'''

window_center = 40
window_width = 80

paths = Path('/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train_images') # set the path to where your ~/stage_1_train_images folder is
pathsTest = Path('/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_test_images') # set the path to where your ~/stage_1_test_images folder is
trainCSVpath = '/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train.csv'
revisedtrainCSVpath = '/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train_revised.csv'

path = list(paths.iterdir())[0]
im = pydicom.read_file(str(path))

def window_and_normalize(im):
    rescaled = im.pixel_array * float(im.RescaleSlope) + float(im.RescaleIntercept)
    windowed = rescaled.clip(min=window_center-window_width, max=window_center+window_width)

    return (windowed + np.negative(window_center-window_width)) / (window_width * 2 * 1/255)

plt.imshow(window_and_normalize(im), cmap=plt.cm.bone)

def resize(src, dst, sz):
    im = pydicom.read_file(str(src))
    ary = window_and_normalize(im)
    im = PIL.Image.fromarray(ary.astype(np.int8), mode='L')
    im.resize((sz,sz), resample=PIL.Image.BICUBIC).save(f'{dst}/{src.stem}.png')
    
import pandas as pd
df = pd.read_csv('/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train.csv') # path to your CSV
df[df.ID.str.match('ID_6431af929')]
df = df[~df.ID.str.match('ID_6431af929')]
df.to_csv(revisedtrainCSVpath, index=False)

    
print('Processing Train Set')
#def resize_112(path, _): resize(path, '/media/docear/My Passport/Kaggle/Hemorrhage/data/112/train', 112) # set the destination file address to a new folder that will accept the processed training set
def resize_112(path, _): resize(path, '/home/docear/trainDataKaggle', 112) # set the destination file address to a new folder that will accept the processed training set
parallel(resize_112, list(paths.iterdir())[674257:], max_workers=12)

print('Processing Test Set')
def resize_112_test(path, _): resize(path, '/media/docear/My Passport/Kaggle/Hemorrhage/data/112/test', 112) # set the destination file address to a new folder that will accept the processed test set
parallel(resize_112_test, list(pathsTest.iterdir()), max_workers=12)