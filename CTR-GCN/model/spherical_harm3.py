import math
from multiprocessing.sharedctypes import Value
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.special


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())

        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        
        return y
        
class symmetry_module(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(symmetry_module, self).__init__()
        self.time_channels = 16
        self.emb1 = nn.Sequential(
            nn.BatchNorm2d(in_channels), # try 1 d
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride= stride),
            nn.ReLU(),
        )

        self.fnn = nn.Linear(64,64)

        self.emb2 = nn.Sequential(
            nn.BatchNorm2d(in_channels), # try 1 d
            nn.Conv2d(in_channels, in_channels*5, kernel_size = (3,1), stride= 1),
            nn.ReLU(),
        )

        self.emb3 = nn.Sequential(
            nn.BatchNorm2d(in_channels*5), # try 1 d
            nn.Conv2d(in_channels*5, out_channels, kernel_size = (3,1), stride= (4,1), padding = (1,0)),
            nn.ReLU(),
        )
  
            ## convert euler coordinates (3D coordinates) to spherical coordinates
    def Spherical_trans(self,x):
        ptsnew = torch.zeros(x.shape).cuda(x.get_device())
        xy = x[:, 0] ** 2 + x[:, 1] ** 2
        r = torch.sqrt(xy + x[:, 2] ** 2)
        theta = torch.atan2(torch.sqrt(xy)+0.00001, x[:, 2]+0.00001)  # azimuthal: for elevation angle defined from Z-axis down -> INTERESTING
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        phi = torch.atan2(x[:, 1]+0.00001, x[:, 0]+0.00001) # polar
        return r, theta, phi

    def Spherical_harm(self, l_range, theta, phi,x):
        theta = theta.cpu().detach()
        phi = phi.cpu().detach()
        result = None
        #test1 = []
        #torch.pi = torch.acos(torch.zeros(1)).item() * 2
        for l in range(l_range+1):
            m_range = np.arange(0,l+1,1, dtype=int)
            for m in m_range:
                test = scipy.special.sph_harm(m, l, theta, phi, out=None)
                test = torch.tensor(test).unsqueeze(1).cuda(x.get_device())
                result = torch.cat((result, test),dim =1) if result is not None else test
      
                #test1.append((l,m))
                #norm =np.sqrt((2*l+1)* math.factorial((l-m)/4*torch.pi*math.factorial(l+m)))

        #raise ValueError(test.shape, result.shape, test1)
        return result

    def forward(self, x):
        N, C, T, V = x.size()

        # convert from catesian coordinates to cylindrical
        pol_input = x#.view(N,-1) # [128, 3, 64, 25]
        r, theta, phi = self.Spherical_trans(pol_input) # each [128, 64, 25]
        spher_harm = self.Spherical_harm(3, theta, phi,x) # [128, 10, 64, 25]
        phase = spher_harm.angle()
        spher_harm_tran = torch.transpose(phase, 2,3)
        test = x[0,0]
        raise ValueError(spher_harm.shape, pol_input.shape, r.shape, theta.shape, phi.shape, test.shape)
        
        check = spher_harm_tran @ phase
        sym1,_ = torch.max(check, dim = 1, keepdim= True) # tried argsort but couldnt figure it out
        #sym1 = torch.tensor(sym1)
        sym2 = torch.argmax(sym1, dim = 3, keepdim= True) # tried argsort but couldnt figure it out
        # sym = torch.argsort(check_mag, dim = 3, descending = True)
        unique, counts = torch.unique(sym2, return_counts = True, dim = 2, sorted= False)
        sym3 = sym2 * (counts-1)


        raise ValueError(unique.shape, counts[0], sym2[0],sym3[0])     
        #x = pol.view(N,C,T,V)

        
        f = self.emb1(theta)
        #raise ValueError(theta.shape, x.shape)
        #f = torch.fft.fft(x,dim=2)
        #f = f.angle() # retrieve phase
        
        f = f.permute(0,1,3,2).contiguous() # N,C,V,T
        f = self.fnn(f) # re-arrange frequencies
        f = f.permute(0,1,3,2).contiguous() # N,C,T,V -> 64 channels
        #raise ValueError(theta.shape, f.shape)
        f = self.emb2(f) # decrease time dim
        
        
        
        f = self.emb3(f)
        # raise ValueError(theta.shape, f.shape)
        return f


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class final_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(final_TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.embed = nn.Conv2d(in_channels, out_channels, 1)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x,f):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        #raise ValueError(y.shape, f.shape)
        y = y+f
        y = self.embed(y)

        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0) #create 3 times identity matrix and stack them into 3D array, matching input dims -> when adaptive = TRUE: learnable
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.sym = symmetry_module(1,256)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = final_TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        N, C, T, V, M = x.size()

        # print(x.shape) -> 64, 3, 64, 25, 2
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # order is now N,(M,V,C),T
        #print(x.shape) -> 64, 150, 64
        x = self.data_bn(x)
        #print(x.shape) -> shape stays the same
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # x is now 4 D: N*M, C, T,V
        # print(x.shape) -> 128, 3, 64, 25
        f = self.sym(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x,f)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
   
        return self.fc(x)
