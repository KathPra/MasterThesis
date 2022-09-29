# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F
import scipy.special
from numpy import linalg as LA


cset = np.load('NTU120_CSet.npz')
x_train = cset["x_train"]
y_train = cset["y_train"]
x_test = cset["x_test"]
y_test = cset["y_test"]

print(x_test.shape)
raise ValueError

def valid_crop_resize(data_numpy,valid_frame_num,p_interval=[0.75],window=64):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        # some probability between 0.5 and 1
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    # bilinear mean linear in 2D space
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    # (C*V*M) x T=window
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    return data

def Spherical_coord(x):
    colatitude = np.arctan2(np.sqrt(x[:, 0]**2+ x[:, 1]**2),x[:, 2])
    azimuth = np.arctan2(x[:, 1], x[:, 0])
    p = np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) # magnitude of vector
    x = np.stack([p, azimuth, colatitude],axis=1)
    
    return x

def Spherical_harm(x,l_range):
    result = None
    #test1 = []
    #torch.pi = torch.acos(torch.zeros(1)).item() * 2
    for l in range(l_range+1):
        m_range = np.arange(-l,l+1,1, dtype=int)
        for m in m_range:
            test = scipy.special.sph_harm(m, l, x[:,1],x[:,2], out=None) # theta: azimuth, phi = colatitude
            test = np.expand_dims(test, axis=1)
            result = np.concatenate((result, test),axis =1) if result is not None else test
            # result: torch.Size([128, 21, 64, 25, 25]))
    
    # raise ValueError(test.shape, result.shape)
    return result


# Copied from original feeder file
valid_frame_num = np.sum(x_train.sum(0).sum(-1).sum(-1) != 0)

N,T,_ = x_train.shape
x_train = x_train.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
x_tran = np.ones((x_train.shape[0], 3,64,25,2 ))

for i in range(0,x_train.shape[0]): # slight modification to iterate through all the samples
    x_tran[i] = valid_crop_resize(x_train[i], valid_frame_num)

print(x_tran.shape)

x_spher = None
batch_size = 68
for batch in range(0,801,1):
    start = batch * 68
    end =start + 68
    data = x_tran[start:end]
    data = Spherical_coord(data)
    data = Spherical_harm(data, 8)
    print(data.shape)
    data = np.absolute(data)
    print("absolute value",data.shape)
    x_spher = np.concatenate((x_spher, data), axis = 0) if x_spher is not None else data
print(x_spher.shape)



# print("data loaded")
print(x_train.shape)
#print(y_train.shape)
print(x_test.shape)
# print(y_test.shape)

np.savez("NTU120_CSet_SH.npz", x_spher, y_train, x_test, y_test)