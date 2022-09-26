from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# load data
folder ="/ceph/lprasse/MasterThesis/CTR-GCN/work_dir/ntu120/"

## extract relevant data from log file
def extract_data(file, loss, acc):
    linenum = 0
    for line in file:
        linenum += 1
        if np.char.find(line, "Mean test loss")> 0:    # if case-insensitive match,
            start = np.char.find(line, "Mean test loss") + 31
            loss.append(line[start:start + 10 ])
        if np.char.find(line, "Top1:")> 0:    # if case-insensitive match,
            start = np.char.find(line, "Top1:") + 6
            acc.append(line[start:start + 5])
    loss = np.asarray(loss, dtype=float)
    acc = np.asarray(acc, dtype=float) / 100

    if len(loss)> 65:
        loss = loss[-65:]
    
    if len(acc)> 65:
        acc = acc[-65:]
    
    return loss, acc

## Plot
def plot_loss_acc(loss1, loss2, top1acc1, top1acc2, name, save_name):
        # Plot loss
    epoch = np.arange(0,65,1)
    
    if len(loss1)< 65:
        epoch = np.arange(0,len(loss1),1)
    plt.plot(epoch, loss1, label ="Manual Normalization after BN")
    epoch = np.arange(0,65,1)
    if len(loss2)< 65:
        epoch = np.arange(0,len(loss2),1)
    plt.plot(epoch, loss2, label ="Manual Normalization before BN")
    plt.title(f'Effect of BN Position on Loss: {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_BN_effect_loss.png")
    plt.close()

    epoch = np.arange(0,65,1)
    if len(top1acc1)< 65:
        epoch = np.arange(0,len(top1acc1),1)
    plt.plot(epoch, top1acc1, label ="Manual Normalization after BN")
    epoch = np.arange(0,65,1)
    if len(top1acc2)< 65:
        epoch = np.arange(0,len(top1acc2),1)
    plt.plot(epoch, top1acc2, label ="Manual Normalization before BN")
    plt.title(f'Effect of BN Position on Top 1 Accuracy: {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Top 1 Accuracy')
    plt.legend()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_BN_effect_acc.png")
    plt.close()


## AZIMUTH
# Azimuth (cosine sim) after BN
file1="csub/azimuth_afterBN/log.txt"
experiments1 = genfromtxt(folder+file1, delimiter='\n',dtype = str)
loss1 = []                       # The list where we will store results.
top1acc1 = []
loss1, top1acc1 = extract_data(experiments1, loss1, top1acc1)


# Azimuth (cosine sim) after BN
file2="csub/azimuth_cent/log.txt"
experiments2 = genfromtxt(folder+file2, delimiter='\n',dtype = str)
loss2 = []                       # The list where we will store results.
top1acc2 = []
loss2, top1acc2 = extract_data(experiments2, loss2, top1acc2)

plot_loss_acc(loss1, loss2, top1acc1, top1acc2, "Azimuth","Azimuth")

## LOCAL SHT w/ l=2
# local SHT w/ l = 2 after BN
file1="csub/local_SHT3_1/log.txt"
experiments1 = genfromtxt(folder+file1, delimiter='\n',dtype = str)
loss1 = []                       # The list where we will store results.
top1acc1 = []
loss1, top1acc1 = extract_data(experiments1, loss1, top1acc1)


# local SHT w/ l = 2 before BN
file2="csub/local_SHT3/log.txt"
experiments2 = genfromtxt(folder+file2, delimiter='\n',dtype = str)
loss2 = []                       # The list where we will store results.
top1acc2 = []
loss2, top1acc2 = extract_data(experiments2, loss2, top1acc2)

plot_loss_acc(loss1, loss2, top1acc1, top1acc2, "Local SHT w/ l=2", "Local_SHT_l2")
