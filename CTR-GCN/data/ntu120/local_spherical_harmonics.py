# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn.functional as F
import scipy.special

def valid_crop_resize(data_numpy,valid_frame_num,p_interval=[0.75],window=64):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    p = p_interval[0]
    bias = int((1-p) * valid_size/2)
    data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
    cropped_length = data.shape[1]

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    # bilinear mean linear in 2D space
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    # (C*V*M) x T=window
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    return data

def local_coord(x):
    _, _, _, V = x.size()
    x = np.stack([x] * V, axis=4) - np.stack([x] * V, axis=3)
    return x

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

def transform_data(x):
    # Copied from original feeder file
    valid_frame_num = np.sum(x.sum(0).sum(-1).sum(-1) != 0)
    N,T,_ = x.shape
    x = x.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    x_tran = np.ones((x.shape[0], 3,64,25,2 ))

    for i in range(0,x.shape[0]): # slight modification to iterate through all the samples
        x_tran[i] = valid_crop_resize(x[i], valid_frame_num)

    print(x_tran.shape)
    x_loc = local_coord(x_tran)
    print(x_loc.shape)

    x_spher = None
    batch_size = 68
    for batch in range(0,801,1):
        start = batch * 68
        end =start + 68
        if end > x_tran.shape[0]:
            end = x_tran.shape[0]
        data = x_tran[start:end]
        data = Spherical_coord(data)
        data = Spherical_harm(data, 8)
        data = np.absolute(data)
        x_spher = np.concatenate((x_spher, data), axis = 0) if x_spher is not None else data
        if batch%10 ==0:
            print(x_spher.shape) 
    print(x_spher.shape)
    return x_spher

def reshape_data(x):
    x = torch.tensor(x,dtype=torch.float)
    N, _, T, V,_ = x.size()
    x = x.permute(0,1,4,2,3).contiguous().view(N,-1,T,V).contiguous().numpy()
    return x
        

## CSET
cset = np.load('NTU120_CSet.npz')
x_train = cset["x_train"]
y_train = cset["y_train"]
x_test = cset["x_test"]
y_test = cset["y_test"]

print(x_train.shape)
x_train = transform_data(x_train)
print(x_train.shape)
x_train = reshape_data(x_train)
print(x_train.shape)

print(x_test.shape)
x_test = transform_data(x_test)
print(x_test.shape)
x_test= reshape_data(x_test)
print(x_test.shape)

np.savez("NTU120_CSet_LSH.npz", x_train, y_train, x_test, y_test)

## CSUB
cset = np.load('NTU120_CSub.npz')
x_train = cset["x_train"]
y_train = cset["y_train"]
x_test = cset["x_test"]
y_test = cset["y_test"]

print(x_train.shape)
x_train = transform_data(x_train)
print(x_train.shape)
x_train = reshape_data(x_train)
print(x_train.shape)

print(x_test.shape)
x_test = transform_data(x_test)
print(x_test.shape)
x_test= reshape_data(x_test)
print(x_test.shape)

np.savez("NTU120_CSub_LSH.npz", x_train, y_train, x_test, y_test)
