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


cset = np.load('NTU120_CSet.npz')
x_train = cset["x_train"]
y_train = cset["y_train"]
x_test = cset["x_test"]
y_test = cset["y_test"]

def valid_crop_resize(data_numpy,valid_frame_num,p_interval=[0.5, 1],window=64):
    # input: C,T,V,M
    N, C, T, V, M = data_numpy.shape
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
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

valid_frame_num = np.sum(x_train.sum(0).sum(-1).sum(-1) != 0)
print(x_train.shape)
N,T,_ = x_train.shape
x_train = x_train.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
print(x_train.shape)
x_train = valid_crop_resize(x_train, valid_frame_num)

print("data loaded")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


print(np.amin(x_train, axis=(0,1,2)))
print(np.amax(x_train, axis = (0,1,2)))



print(x_train.shape)
print(type(x_train))
print(np.sum(x_train.sum(0).sum(-1).sum(-1) != 0))
