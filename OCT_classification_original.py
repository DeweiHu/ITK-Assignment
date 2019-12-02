# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:01:51 2019

@author: hudew
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Train data/label loading from directory 
dir_normal = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\normal_train'
list_normal = []
for filename in os.listdir(dir_normal):
    if filename.endswith(".jpeg"):
        list_normal.append(os.path.join(dir_normal,filename))
    else:
        continue
GT_normal = np.zeros([len(list_normal),1]).astype(np.uint8)

dir_cnv = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\cnv_train'
list_cnv = []
for filename in os.listdir(dir_cnv):
    if filename.endswith(".jpeg"):
        list_cnv.append(os.path.join(dir_cnv,filename))
    else:
        continue
GT_cnv = np.ones([len(list_cnv),1]).astype(np.uint8)

dir_dme = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\dme_train'
list_dme = []
for filename in os.listdir(dir_dme):
    if filename.endswith(".jpeg"):
        list_dme.append(os.path.join(dir_dme,filename))
    else:
        continue
GT_dme = (np.ones([len(list_dme),1])*2).astype(np.uint8)
    
dir_drusen = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\drusen_train'
list_drusen = []
for filename in os.listdir(dir_drusen):
    if filename.endswith(".jpeg"):
        list_drusen.append(os.path.join(dir_drusen,filename))
    else:
        continue
GT_drusen = (np.ones([len(list_drusen),1])*3).astype(np.uint8)

global data_train, label_train
data_train = list_normal+list_cnv+list_dme+list_drusen
label_train = np.vstack([GT_normal,GT_cnv,GT_dme,GT_drusen])

del list_cnv,list_dme,list_drusen,list_normal,GT_cnv,GT_dme,GT_drusen,GT_normal

#%% Test data/label loading from directory
dir_normal = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\normal_test'
list_normal = []
for filename in os.listdir(dir_normal):
    if filename.endswith(".jpeg"):
        list_normal.append(os.path.join(dir_normal,filename))
    else:
        continue
GT_normal = np.zeros([len(list_normal),1]).astype(np.uint8)

dir_cnv = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\cnv_test'
list_cnv = []
for filename in os.listdir(dir_cnv):
    if filename.endswith(".jpeg"):
        list_cnv.append(os.path.join(dir_cnv,filename))
    else:
        continue
GT_cnv = np.ones([len(list_cnv),1]).astype(np.uint8)

dir_dme = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\dme_test'
list_dme = []
for filename in os.listdir(dir_dme):
    if filename.endswith(".jpeg"):
        list_dme.append(os.path.join(dir_dme,filename))
    else:
        continue
GT_dme = (np.ones([len(list_dme),1])*2).astype(np.uint8)
    
dir_drusen = r'C:\Users\hudew\OneDrive\桌面\OCT_1536_496\drusen_test'
list_drusen = []
for filename in os.listdir(dir_drusen):
    if filename.endswith(".jpeg"):
        list_drusen.append(os.path.join(dir_drusen,filename))
    else:
        continue
GT_drusen = (np.ones([len(list_drusen),1])*3).astype(np.uint8)

global data_test, label_test
data_test = list_normal+list_cnv+list_dme+list_drusen
label_test = np.vstack([GT_normal,GT_cnv,GT_dme,GT_drusen])

del list_cnv,list_dme,list_drusen,list_normal,GT_cnv,GT_dme,GT_drusen,GT_normal

#%% Dataset setup
import matplotlib.image as mpimg
import torch
import torch.utils.data as Data
import nibabel as nib


EPOCH = 30
BATCH_SIZE = 1
LR = 0.0001

class Train_Dataset(Data.Dataset):
    
# transform numpy array to tensor
    def transform(self, data,label):
        x_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        y_tensor = torch.from_numpy(label).type(torch.LongTensor)
        return x_tensor,y_tensor
    
    def __init__(self):
        self.pair = ()
        self.num = len(data_train)
       # match data-label pairs
        for i in range(self.num):
            x = mpimg.imread(data_train[i])
            y = np.squeeze(label_train[i],axis=0)
            self.pair = self.pair+((x,y),)
    
    def __len__(self):
        return self.num
       
    def __getitem__(self,idx):
        (img,gt) = self.pair[idx]
        x_tensor,y_tensor = self.transform(img,gt)
        return x_tensor,y_tensor
    
train_loader = Data.DataLoader(dataset=Train_Dataset(),batch_size=BATCH_SIZE,shuffle=True)

#%%
class Test_Dataset(Data.Dataset):
    
# transform numpy array to tensor
    def transform(self, data,label):
        x_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        return x_tensor,label
    
    def __init__(self):
        self.pair = ()
        self.num = len(data_test)
       # match data-label pairs
        for i in range(self.num):
            x = mpimg.imread(data_test[i])
            y = label_test[i,0]
            self.pair = self.pair+((x,y),)
    
    def __len__(self):
        return self.num
       
    def __getitem__(self,idx):
        (img,gt) = self.pair[idx]
        x_tensor,y = self.transform(img,gt)
        return x_tensor,y
    
test_loader = Data.DataLoader(dataset=Test_Dataset(),batch_size=BATCH_SIZE,shuffle=False)


#%% Archaetecture
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # [496,1536]
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, 
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # [298,768]
        self.stage_2 = nn.Sequential(
             nn.Conv2d(in_channels=16, out_channels=32,
                       kernel_size=5, stride=1, padding=2),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2)
        )
        # [149,384]
        self.stage_3 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=64,
                       kernel_size=5, stride=1, padding=2),
             nn.ReLU(),
             nn.AdaptiveMaxPool2d((74,192))
        )
        self.fc1 = nn.Linear(74*192*64,4)
        
    def forward(self,x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = x.view(x.size(0),-1)
        output = self.fc1(x)
        return output

model = CNN()

model.cuda()

#%% Run the model
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import time
from scipy.special import softmax

optimizer = torch.optim.RMSprop(model.parameters(),lr=LR)
scheduler = StepLR(optimizer, step_size=2, gamma=0.2)
loss_func = nn.CrossEntropyLoss()

# Converge
Loss = []
predict = np.zeros([365,1]).astype(np.uint8)

#Iterations
t_start = time.time()

for epoch in range(EPOCH):
    for step,[x,y] in enumerate(train_loader):
        train_x = Variable(x).cuda()
        train_y = Variable(y).cuda()
        pred_y = model(train_x)
        # loss and backward propagation
        loss = loss_func(pred_y, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            loss_np = torch.tensor(loss.cpu()).numpy()
            Loss.append(loss_np)
            for num,[x,y] in enumerate(test_loader):
                test_x = Variable(x).cuda()
                pred = model(test_x)
                pred = torch.tensor(pred).cpu().numpy()
                pred = np.argmax(softmax(pred))
                predict[num,0] = pred
            
            dist = predict-label_test
            
            test_accuracy = np.count_nonzero(dist==0)/365
                
            print('epoch:',epoch,'| Iter:',step,'| Training loss:',loss_np, '| Test Accuracy:', test_accuracy)
            print('------------------------------------------------------------------------------------------')

t_end = time.time()
print((t_end-t_start)/60,'min')