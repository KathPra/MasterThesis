import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,in_channels=3, seg = 64, graph=None, graph_args=dict(),  drop_out=0, adaptive=True, num_set=3): 
        # not needed: graph=None, graph_args=dict(),  drop_out=0, adaptive=True, num_set=3 
        super(Model, self).__init__()

        self.N = 128
        self.seg = seg
        self.in_channels = in_channels
        self.num_point = num_point

        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * num_point)

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=True)
        self.spa_embed = embed(num_point, 64, norm=False, bias=True)
        self.joint_embed = embed(3, 64, norm=True, bias=True)
        self.dif_embed = embed(3, 64, norm=True, bias=True)
        self.fourier_embed = embed(3, 64, norm=True, bias=True)
        self.sym_embed = embed(3, 64, norm=True, bias=True)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.N*2, self.N * 4, bias=True)
        self.compute_g1 = compute_g_spa(self.N, self.N*2, bias=True)
        self.gcn1 = gcn_spa(self.N, self.N, bias=True)
        self.gcn2 = gcn_spa(self.N, self.N*2, bias=True)
        self.gcn3 = gcn_spa(self.N*2, self.N*2, bias=True)
        self.fc = nn.Linear(self.N * 4, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, x):
        N, C, T, V, M = x.size()
        input = x.clone()
        x = x.permute(0,4,3,2,1).contiguous().view(N* M ,T,V,C)
        # shape N*M, T,V,C
        x = x.permute(0, 3, 2, 1).contiguous()
        # shape N*M, C, V, T
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1] # compute velocity by substraction along time axis
        first = dif.new(N*M, dif.size(1), V, 1).zero_()
        dif = torch.cat([first, dif], dim=-1)

        # Input prep
        spa = self.one_hot(N*M, self.num_point, self.seg)
        spa = spa.permute(0, 3, 2, 1).cuda(x.get_device())
        tem = self.one_hot(N*M, self.seg, self.num_point)
        tem = tem.permute(0, 3, 1, 2).cuda(x.get_device())
        
        pos = self.joint_embed(x)
        tem1 = self.tem_embed(tem)
        spa1 = self.spa_embed(spa)
        dif = self.dif_embed(dif)

        dy = pos + dif
        # Joint-level Module
        x= torch.cat([dy, spa1], 1)
        # raise ValueError(x.shape) -> input size: [39, 3, 64, 25, 2]
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        # Frame-level Module
        x = x + tem1
        x = self.cnn(x)
        # Classification
        x = self.maxpool(x)
        x = torch.flatten(x, 1)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)

    def one_hot(self, N, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(N, tem, 1, 1)

        return y_onehot

class norm_data(nn.Module):
    def __init__(self, dim= 64, num_joint = 25):
        super(norm_data, self).__init__()
        self.bn = nn.BatchNorm1d(dim*num_joint) # number of features

    def forward(self, x):
        N, C, V, T = x.size()
        x = x.view(N, -1, T)
        x = self.bn(x) # requires input of form N (batch size),C(# features),L (length)
        x = x.view(N, -1, V, T).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
    
