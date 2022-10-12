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
    plt.savefig(f"vis/ConfusionMatrix_{save_name}.png")
    plt.close()

# load class dict
ntu120_class = np.loadtxt("/ceph/lprasse/MasterThesis/CTR-GCN/vis/ntu120_classes.txt", delimiter='\t',dtype=str)
#class_ind = np.arange(0,120,1)
#class_dict = dict(zip(class_ind, ntu120_class))
#print(class_dict)

# load data
def load_data(file_name):
    folder ="/ceph/lprasse/MasterThesis/CTR-GCN/work_dir/ntu120/"
    file=file_name+"_test_each_class_acc.csv"
    experiments = genfromtxt(folder+file, delimiter=',')
    #print(experiments.shape)
    accuracy = experiments[0] # for each class
    confusion = experiments[1:] # for each class i: true label (row), j: prediction (column)
    return accuracy, confusion

def top_flop(accuracy, class_dict):
    max = np.sort(accuracy)[-5:]
    max = np.flip(max)
    amax = np.argsort(accuracy)[-5:]
    amax = np.flip(amax)
    print(amax)
    max_label = class_dict[amax]
    best_dict = dict(zip(max_label, max))

    min = np.sort(accuracy)[:6]
    amin = np.argsort(accuracy)[:6]
    print(amin)
    min_label = class_dict[amin]
    worst_dict = dict(zip(min_label, min))
    print("Best: ", best_dict, "Worst: ", worst_dict)
    return best_dict, worst_dict

## Baseline (CSET)
print("Baseline SET")
acc, conf = load_data("cset/baseline_imp/epoch64")
plot_CM(conf, "baseline_CSET")
BaselineIMP_t5, BaselineIMP_l5 = top_flop(acc, ntu120_class)

## Baseline
print("Baseline CSUB")
acc, conf = load_data("csub/baseline/epoch57")
plot_CM(conf, "baseline_CSUB")
Baseline_t5, Baseline_l5 = top_flop(acc, ntu120_class)

## Baseline_imp
print("Baseline_imp CSUB")
acc, conf = load_data("csub/baseline_imp/epoch61")
plot_CM(conf, "baseline_imp_CSUB")
BaselineIMP_t5, BaselineIMP_l5 = top_flop(acc, ntu120_class)

## spherical_coord
print("Spherical_coord CSUB")
acc, conf = load_data("csub/spherical_coord/epoch61")
plot_CM(conf, "spherical_coord_CSUB")
SphericalCoord_t5, SphericalCoord_l5 = top_flop(acc, ntu120_class)

## local_sphercoord
print("Local_Sphercoord CSUB")
acc, conf = load_data("csub/local_spher_coord/epoch58")
plot_CM(conf, "local_spherical_coord_CSUB")
SphericalCoord_t5, SphericalCoord_l5 = top_flop(acc, ntu120_class)