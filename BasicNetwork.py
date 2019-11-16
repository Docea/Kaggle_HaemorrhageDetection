#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:42:54 2019

@author: docear
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from random import shuffle
import DatasetHaemo as DH
import torchvision
from torchvision import transforms, datasets
from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator as BSI


train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)

trainPath = Path('../trainDataKaggle/')
trainpath = '../trainDataKaggle/'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.pool = nn.MaxPool2d(2,2) # Defines a type of maxpooling to be used
        self.conv2 = nn.Conv2d(3, 8, 4)
        self.fc1 = nn.Linear(8*26*26, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,8*26*26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x
        #return F.softmax(x,dim=1)
        
net = Net()
 

#df = pd.read_csv('/media/docear/My Passport/Kaggle/Hemorrhage/train_labels_as_strings.csv')
#df.labels.fillna('/media/docear/My Passport/Kaggle/Hemorrhage/train_labels_as_strings.csv',inplace=True)


SETUP = False # Change this variable to triggger 
if SETUP == True:
    # Extracting Training Data (below)
    dfLabels = pd.read_csv('/media/docear/My Passport/Kaggle/Hemorrhage/train_pivot.csv') # csv relating to the training data
    index = list(trainPath.iterdir()) # lists paths to all train images
    posn = len(trainpath) # the length of the prefix to the image name as found in the .csv file
    
    filenames = [] # initialising array to hold all image names (without prefixes)
    
    for idx in range(len(index)): # filling array with image names
        filenames.append(str(index[idx])[posn:])
        
    dfLabels.set_index("fn",inplace=True) # sets the relevant data entry for look-up to be the filename (fn)
    
    nValidation = 32000
    validationFiles = filenames[-nValidation-1:-1]
    filenames = filenames[0:-nValidation-1]
    
    dfLabels = dfLabels['epidural'].to_frame()
    dfLabels = dfLabels.sort_values(by='epidural',ascending=False)
    nNegative = dfLabels['epidural'].value_counts()[0]
    nPositive = dfLabels['epidural'].value_counts()[1]
    
    
    totalPositive = nPositive
    i = 0
    while totalPositive < nNegative:
        for index in dfLabels.index[0:nPositive+1]:
            if index not in validationFiles:
                filenames.append(index)
            i+=1
            if i % 10000 == 0:
                print(str(i)+" / "+str(nNegative-nPositive))
        totalPositive += nPositive
    
    
    
    EPOCHS = 3
    #possibleBatches = round(len(index)/batchsize)


#trainDataset = DH.DatasetHeamo('/media/docear/My Passport/Kaggle/Hemorrhage/train_pivot.csv',filenames,trainpath,[1,112,112])
trainDataset = DH.DatasetHeamo(dfLabels,filenames,trainpath,[1,112,112])
#trainLoader = BSI(trainDataset, labels=dfLabels.epidural, batch_size=32,shuffle=True,batch_balancing=True)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
trainIter = iter(trainLoader)
#criterion = torch.nn.NLLLoss()
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
#optimizer = optim.Adam(net.parameters(), lr=0.1)
feedbackFreq = 1000

validationDataset = DH.DatasetHeamo(dfLabels,validationFiles,trainpath,[1,112,112])
validationLoader = DataLoader(validationDataset, batch_size=32, shuffle=True)



print("Starting Training")

for epoch in range(EPOCHS):
    running_loss = 0.0
    totalP = 0
    for i, data in enumerate(trainLoader,0):
        # get next batch
        inputs, labels = data
        #inputs = torch.stack([item[0] for item in data])
        #labels = torch.stack([item[1] for item in data])
        #labels = labels.type(torch.LongTensor)
        totalP += sum(labels)
        # set cumulative gradients to 0
        optimizer.zero_grad()
        
        # forward pass ; backward pass; optimization
        outputs = net(inputs)
        outputs = outputs.reshape([outputs.size()[0]])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
        if i % feedbackFreq == feedbackFreq-1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / feedbackFreq))
            print('number of positive labels tried: '+str(totalP))
            totalP=0
            running_loss = 0.0
            
            validationLoss = 0.0
            for j, vData in enumerate(validationLoader,0):
                inputs, labels = vData
        
                # forward pass ; backward pass; optimization
                outputs = net(inputs)
                outputs = outputs.reshape([outputs.size()[0]])
                loss = criterion(outputs, labels)
                
                # statistics
                validationLoss += loss.item()
            print('validation loss: ' + str(validationLoss))
        


print('Finished Training')



