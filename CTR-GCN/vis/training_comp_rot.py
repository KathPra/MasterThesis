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
    i = 0
    while i < len(acc):
        if np.char.find(acc[i], "%"):
            acc[i] = acc[i].replace("%", "")
        i += 1
    acc = np.asarray(acc, dtype=float) / 100

    if len(loss)> 65:
        loss = loss[-65:]
    
    if len(acc)> 65:
        acc = acc[-65:]
    
    return loss, acc

def plot_loss(loss_list, loss_labels, save_name):
        # Plot loss
    loss_count = 0
    for i in loss_list:
        epoch = np.arange(0,65,1)
        if len(i)< 65:
            epoch = np.arange(0,len(i),1)
        plt.plot(epoch, i, label =loss_labels[loss_count])
        loss_count+=1 
    plt.title(f'Training loss comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_comp_loss.png")
    plt.close()

def plot_acc(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        epoch = np.arange(0,65,1)
        if len(i)< 65:
            epoch = np.arange(0,len(i),1)
        plt.plot(epoch, i, label =acc_label[acc_count])
        acc_count+=1 
    plt.title(f'Test accuracy comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_comp_accuracy.png")
    plt.close()


## Baseline
file1="csub/baseline/log.txt"
experiments1 = genfromtxt(folder+file1, delimiter='\n',dtype = str)
loss1 = []                       # The list where we will store results.
top1acc1 = []
loss1, top1acc1 = extract_data(experiments1, loss1, top1acc1)


# Baseline Cent
file2="csub/baseline_imp/log.txt"
experiments2 = genfromtxt(folder+file2, delimiter='\n',dtype = str)
loss2 = []                       # The list where we will store results.
top1acc2 = []
loss2, top1acc2 = extract_data(experiments2, loss2, top1acc2)


# Baseline Cent/Rot
file3="csub/baseline_rot/log.txt"
experiments3 = genfromtxt(folder+file3, delimiter='\n',dtype = str)
loss3 = []                       # The list where we will store results.
top1acc3 = []
loss3, top1acc3 = extract_data(experiments3, loss3, top1acc3)





label_list =  ["Baseline","Baseline (Cent.)", "Baseline (Cent., Rot.)"]

plot_loss([loss1, loss2, loss3],label_list, "Rotation")
plot_acc([top1acc1, top1acc2, top1acc3], label_list, "Rotation")
# label_list =  ["Baseline","Colatitude (cosine sim.)","Colatitude (radians)","Colatitude (mathem.)" ]

# plot_loss([loss1, loss2, loss3, loss4],label_list, "colatitude_computation")
# plot_acc([top1acc1, top1acc2, top1acc3, top1acc4], label_list, "colatitude_computation")