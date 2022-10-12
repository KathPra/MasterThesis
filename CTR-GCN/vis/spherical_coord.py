from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# Plot Confusion matrix
def plot_CM(confusion, save_name):
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    fig = plt.imshow(confusion, cmap='hot', interpolation='nearest', origin="upper")
    ax.xaxis.tick_top()
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Actuals', fontsize=15)
    # ax.set_xlim(0,120)
    # ax.set_ylim(120,0)
    plt.title('Confusion Matrix', fontsize=18)
    cbar = plt.colorbar(fig)
    # cbar.solids.set_edgecolor("face")
    # plt.draw()
    plt.savefig(f"vis/{save_name}.png")
    plt.close()

# load class dict
ntu120_class = np.loadtxt("/ceph/lprasse/MasterThesis/CTR-GCN/vis/ntu120_classes.txt", delimiter='\t',dtype=str)
class_ind = np.arange(0,120,1)
class_dict = dict(zip(class_ind, ntu120_class))
#print(class_dict)

# load data
def load_data(file_name):
    folder ="/ceph/lprasse/MasterThesis/CTR-GCN/work_dir/ntu120/"
    file=file_name+"_test_each_class_acc.csv"
    experiments = genfromtxt(folder+file, delimiter=',')
    print(experiments.shape)
    accuracy = experiments[0] # for each class
    confusion = experiments[1:] # for each class i: true label (row), j: prediction (column)
    return accuracy, confusion

## Baseline
acc, conf = load_data("csub/baseline/epoch57")
plot_CM(conf, "baseline")
