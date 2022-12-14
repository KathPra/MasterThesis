from cProfile import label
import math
from multiprocessing.sharedctypes import Value
from re import I
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import linalg as LA
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
    def __init__(self):
        super(symmetry_module, self).__init__()

    def azimuth(self,x):
        N,C,T,V = x.size()
        
        # compute spine -> z-axis for azimuth computation
        spine_vec = x[:,:,:,20]-x[:,:,:,0] 
        norm_spine = (spine_vec / (LA.norm(spine_vec, dim=1).unsqueeze(1))).permute(0,2,1).unsqueeze(2).contiguous().view(N*T,1,C) # N, C,T,1,1

        # compute vector -> base of spine to joint for azimuth computation
        origine = torch.stack([x[:,:,:,0]]*(x.size(3)-1), dim = 3)
        vec_int = x[:,:,:,1:] - origine  # create vector from base of spine to each joint, except first (vector would have length zero)
        vec = torch.cat((spine_vec.unsqueeze(3),vec_int), dim=3)  # first vector would be (0,0,0,0)-> replace by spine
        norm_vec = (vec / (LA.norm(vec, dim=1).unsqueeze(1))).permute(0,2,1,3).contiguous().view(N*T,C,V) # N, C,T,V,1

        # compute angle between vectors: theta = cos^-1 [(a @ b) / |a|*|b|] -> a,b are normalized, thus |a|=|b|=1 -> theta = cos^-1 [(a @ b)]
        angle = norm_spine @ norm_vec
        angle = angle.nan_to_num(nan=0.0)
        angle = angle.view(N,T,V)
        
        return angle

    def local_coord(self,x):
        N, C, T, V = x.size()

        # new dim: 128 x 3 x 64 x 25 x 25
        x = torch.stack([x] * V, axis=4) - torch.stack([x] * V, axis=3)
        #raise ValueError(torch.trace(x[0,0,0]))
        #raise ValueError(x_mod.shape) 
        #raise ValueError(x_loc[0,:,0,:5,:5])
        
        return x
    
    def Spherical_coord(self,x):

        azimuth = torch.atan2(torch.sqrt(x[:, 0]**2+ x[:, 1]**2),x[:, 2])
        colatitude = torch.atan2(x[:, 1], x[:, 0])
        p = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) # magnitude of vector
        x = torch.stack([p, colatitude, azimuth], dim =1)
        
        return x

    def Spherical_harm(self,x,l):
        x_tran = x.cpu().detach()
        result = None
        #test1 = []
        #torch.pi = torch.acos(torch.zeros(1)).item() * 2
        L = np.arange(1, l+1,1, dtype=int)
        
        for l in L:
            M = np.arange(-l, l+1,1, dtype=int)
            #raise ValueError(L,M)
            for m in M:
                test = scipy.special.sph_harm(m, l, x_tran[:,2],x_tran[:,1], out=None) # theta: azimuth, phi = colatitude
                #raise ValueError(test.shape)
                test = test.unsqueeze(1)
                result = torch.cat((result, test),dim =1) if result is not None else test
                # result: torch.Size([128, 21, 64, 25, 25]))
        
        # raise ValueError(test.shape, result.shape)
        return result

    def forward(self, x, l_range):
        N, C, T, _ = x.size()

        # convert from catesian coordinates to cylindrical
        x = self.local_coord(x) # input [128,3,64,25], output [128, 64, 25]
        x = self.Spherical_coord(x)
        if torch.isnan(x).any():
            raise ValueError("NaN found")
        x = self.Spherical_harm(x, l_range)
        x = x.abs().float() # take norm of SHT

        
        N, _, T, V,_ = x.size()
        x = x.permute(0,1,4,2,3).contiguous().view(N,-1,T,V)
        #raise ValueError(x.shape)
        
        #raise ValueError("test successful")
        return x


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
        self.SHT = 8*self.num_point # l1: 3, l2: 5, l3: 7, l4: 9, l5:11
        self.data_bn = nn.BatchNorm1d(num_person * num_point  * self.SHT) # number of spherical harmonics, 2 * V because of lokal environment for each joint

        self.sym = symmetry_module()
        self.l1 = TCN_GCN_unit(self.SHT, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def plot(self,t,x, dim, string1="Missing"):
        for i in range(20):
            if dim ==5:
                x_val = x[i,0,t,:,0].cpu().detach().numpy()
                y_val = x[i,1,t,:,0].cpu().detach().numpy()
                z_val = x[i,2,t,:,0].cpu().detach().numpy()
            else:
                x_val = x[i,0,t,:].cpu().detach().numpy()
                y_val = x[i,1,t,:].cpu().detach().numpy()
                z_val = x[i,2,t,:].cpu().detach().numpy()
            labels = np.array([0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,1,1,1,1]) #0 for spine, 1 for arms incl. shoulder, 2 for legs incl. hips
            label_dict = {0:"Spine", 1:"Arm", 2:"Leg"}

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for g in np.unique(labels):
                j = np.where(labels == g)
                ax.scatter(x_val[j], y_val[j], z_val[j], label=label_dict[g]) 
 
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1,1)
            ax.set_xticks([-1,-0.5,0,0.5,1])
            ax.set_yticks([-1,-0.5,0,0.5,1])
            ax.set_zticks([-1,-0.5,0,0.5,1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend()
            #ax.legend(labels, ["Spine","Spine","Spine","Spine","Arm","Arm","Arm","Arm","Arm","Arm","Arm","Arm","Leg","Leg","Leg","Leg","Leg","Leg","Leg","Leg","Spine","Arm","Arm","Arm"])
            plt.savefig(f"vis/{i}/{string1}_{t}.png")
            plt.close()

    def lin_trans_angle(self,x):
        # x has dim N x C x T x V
        N,C,T,V = x.size()
        spine_vec = x[:,:,:,20]-x[:,:,:,0] 
        norm_vec = spine_vec / (LA.norm(spine_vec, dim=1).unsqueeze(1)) # shape: N x C x T
        
        ## rotate into yz plane by rotation around z axis
        cos_theta1 = norm_vec[:,0] / torch.sqrt(norm_vec[:,0]**2 + norm_vec[:,1]**2)
        sin_theta1 = norm_vec[:,1] / torch.sqrt(norm_vec[:,0]**2 + norm_vec[:,1]**2)
        first = torch.stack((cos_theta1,sin_theta1,torch.zeros(cos_theta1.shape).cuda(x.get_device())),dim =1)
        second = torch.stack((-sin_theta1, cos_theta1,torch.zeros(cos_theta1.shape).cuda(x.get_device())), dim =1)
        third = torch.stack((torch.zeros(cos_theta1.shape),torch.zeros(cos_theta1.shape),torch.ones(cos_theta1.shape)), dim = 1).cuda(x.get_device())
        rot_z = torch.stack((first,second,third), dim = 1).float()
        
        norm_vec = norm_vec.permute(0,2,1).unsqueeze(3).contiguous().view(N*T,C,1)
        rot_z = rot_z.permute(0,3,1,2).contiguous().view(N*T,C,3)
        x_rotz = rot_z @ norm_vec # unit length
        x_rotz = x_rotz.view(N,T,C,1).permute(0,2,3,1).squeeze()

        ## rotate onto z axis by rotating around the y axis
        cos_theta2 = x_rotz[:,2] / torch.sqrt(x_rotz[:,2]**2 + x_rotz[:,0]**2)
        sin_theta2 = x_rotz[:,0] / torch.sqrt(x_rotz[:,2]**2 + x_rotz[:,0]**2)
        fir = torch.stack((cos_theta2, torch.zeros(cos_theta2.shape).cuda(x.get_device()), -sin_theta2), dim = 1)
        sec = torch.stack((torch.zeros(cos_theta2.shape),torch.ones(cos_theta2.shape),torch.zeros(cos_theta2.shape)), dim = 1).cuda(x.get_device())
        thir = torch.stack((sin_theta2, torch.zeros(cos_theta2.shape).cuda(x.get_device()), cos_theta2), dim =1)
        rot_y = torch.stack((fir,sec,thir), dim = 1).float()
        
        #raise ValueError(norm_vec.shape, x.shape, spine_vec.shape, LA.norm(spine_vec, dim=1).shape)
        x_rotz = x_rotz.permute(0,2,1).unsqueeze(3).contiguous().view(N*T,C,1) # for validation of spine rotation
        rot_y = rot_y.permute(0,3,1,2).contiguous().view(N*T,C,3)
        x_rotzy =  rot_y @ x_rotz # unit length 
        x_rotzy = x_rotzy.view(N,T,C,1).permute(0,2,3,1).squeeze()
        
        return rot_z, rot_y

    def forward(self, x):
        N, C, T, V, M = x.size()

        # Prepare data for local SHT
        x = x.permute(0, 4, 1, 2,3).contiguous().view(N * M, C, T, V)


        # send data to symmetry module
        sym = self.sym(x, 2) # l
        x = sym.cuda(x.get_device())
        _, C, T, V = x.size()

        #raise ValueError(x.shape) #-> 128, 1, 64, 25
        
        # Code from original paper
        #raise ValueError(x.shape)
        x = x.view(N,M,C,T,V).permute(0, 1, 4, 2, 3).contiguous().view(N, M * V * C, T)
        
        #raise ValueError(x.shape)
        # order is now N,(M,V,C),T
        #print(x.shape) -> 64, 150, 64
        x = self.data_bn(x)
        #print(x.shape) -> shape stays the same
        x = x.view(N, M, V,C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # x is now 4 D: N*M, C, T,V
        # print(x.shape) -> 128, 3, 64, 25
        #raise ValueError(x[0,:,0,:])
        #self.plot(0, x, dim = 4, string1="afterBN")
        
        # check values
        first = x[:,0]
        sec = x[:,1]
        third = x[:,2]
        fourth = x[:,3]
        fifth = x[:,4]

        raise ValueError("first", torch.min(first).item(),torch.max(first).item(),torch.mean(first).item(), "fourth", torch.min(fourth).item(),torch.max(fourth).item(),torch.mean(fourth).item(), "fifth", torch.min(fifth).item(),torch.max(fifth).item(),torch.mean(fifth).item())       

        

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
   
        return self.fc(x)
