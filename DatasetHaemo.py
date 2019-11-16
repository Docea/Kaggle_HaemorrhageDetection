#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:26:38 2019

@author: docear
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.image as mpimg


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
        
class DatasetHeamo(Dataset):
    #def __init__(self, csvPath, filenames, trainPath, inDims, transform=None):
    def __init__(self, data, filenames, trainPath, inDims, transform=None):
        self.inChannels = inDims[0]
        self.width = inDims[1]
        self.height = inDims[2]
        #self.data = pd.read_csv(csvPath)
        #self.data.set_index("fn",inplace=True) # sets the relevant data entry for look-up to be the filename (fn)
        self.data = data
        self.filenames = filenames
        self.trainPath = trainPath
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
        #return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = torch.Tensor(mpimg.imread(self.trainPath+self.filenames[index]).reshape((self.inChannels, self.width, self.height)))
        label = torch.Tensor(self.data.loc[self.filenames[index]].values)
        #label = label.long()   # converts to long
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    