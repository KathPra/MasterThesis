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
    accuracy = np.around(accuracy, decimals = 3)
    acc_list = zip(class_dict, accuracy)
    acc = sorted(acc_list, key = lambda acc_list: acc_list[1])
    sorted_dict = dict(acc)

    max = np.flip(accuracy[-5:])
    amax = np.flip(np.argsort(accuracy)[-5:]) 
    max_label = class_dict[amax]
    best_dict = dict(zip(max_label, max))

    min = np.sort(accuracy)[:5]
    amin = np.argsort(accuracy)[:5]
    print(amin)
    min_label = class_dict[amin]
    worst_dict = dict(zip(min_label, min))
    print("Best: ", best_dict, "Worst: ", worst_dict)
    return best_dict, worst_dict, sorted_dict

def plot_acc(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        epoch = np.arange(0,120,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    plt.xlim(left=-0.25, right=120)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_class_accuracy.png")
    plt.close()


def plot_acc_extr(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        i = i[40:66]
        epoch = np.arange(40,66,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    #plt.xlim(left=60, right=81)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_class_accuracy_extract2.png")
    plt.close()

def plot_acc_extr1(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        i = i[58:80]
        epoch = np.arange(58,80,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    #plt.xlim(left=60, right=81)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_class_accuracy_extract2.png")
    plt.close()

def plot_acc_extr2(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        i = i[:30]
        epoch = np.arange(0,30,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    #plt.xlim(left=60, right=81)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/ceph/lprasse/MasterThesis/CTR-GCN/vis/{save_name}_class_accuracy_extract2.png")
    plt.close()

def model_comp(acc1, acc2, class_dict,save_name):
    diff1 = acc1-acc2
    diff1 = np.around(diff1, decimals = 3)
    diff1 = zip(class_dict, diff1)
    diff1 = sorted(diff1, key = lambda diff1: diff1[1])
    # diff2 = acc2-acc1
    # diff2 = np.around(diff2, decimals = 3)
    # diff2 = zip(class_dict, diff2)
    # diff2 = sorted(diff2, key = lambda diff2: diff2[1])

    f = open("./vis/class_comp.txt", "a")
    f.write(save_name+": first is better"+"\n")
    f.writelines(str(item)+"\n" for item in diff1 )
    f.write(save_name+": second is better"+"\n")
    #f.writelines(str(item) for item in diff2 )
    f.close()




## Baseline (CSET)
print("Baseline CSUB")
acc1, conf1 = load_data("csub/baseline/epoch57")
plot_CM(conf1, "baseline_CSUB")
BaselineIMP_t5, BaselineIMP_l5, _ = top_flop(acc1, ntu120_class)

## Mag
print("Mag")
acc2, conf2 = load_data("csub/global_SHT2/epoch62")
plot_CM(conf2, "Mag")
Baseline_t5, Baseline_l5, Baseline = top_flop(acc2, ntu120_class)

## Phase
print("Phase")
acc3, conf3 = load_data("csub/global_SHT2c/epoch62")
plot_CM(conf3, "Phase")
BaselineIMP_t5, BaselineIMP_l5,BaselineIMP = top_flop(acc3, ntu120_class)

## Mag. & Phase
print("Mag. & Phase")
acc4, conf4 = load_data("csub/global_SHT2a/epoch62")
plot_CM(conf4, "Mag. & Phase")
SphericalCoord_t5, SphericalCoord_l5, SphericalCoord = top_flop(acc4, ntu120_class)

## Real & Imag
print("Real & Imag.")
acc5, conf5 = load_data("csub/global_SHT2b/epoch62")
plot_CM(conf5, "Real & Imag.")
SphericalCoord1_t5, SphericalCoord1_l5,SphericalCoord1 = top_flop(acc5, ntu120_class)

## Real
print("Real")
acc6, conf6 = load_data("csub/global_SHT2d/epoch62")
plot_CM(conf6, "Real & Imag.")
SphericalCoord6_t5, SphericalCoord6_l5,SphericalCoord6 = top_flop(acc6, ntu120_class)

## Imag
print("Imag.")
acc7, conf7 = load_data("csub/global_SHT2e/epoch62")
plot_CM(conf6, "Real & Imag.")
SphericalCoord7_t5, SphericalCoord7_l5,SphericalCoord7 = top_flop(acc7, ntu120_class)



results = [Baseline, BaselineIMP, SphericalCoord, SphericalCoord1 ]

# for i in results:
#     print("start")
#     print(i['A27. jump up.'])
#     print(i["A42. staggering."])
#     print(i['A97. arm circles.'])
#     print(i['A113. cheers and drink.'])
#     print(i['A15. take off jacket.'])


## Plot accuracies of models
label_list =  ["Mag.","Phase","Mag. & Phase"]
label_list1 =  ["Real", "Imag.","Real & Imag."]
label_list2 =  ["Mag. & Phase","Real & Imag."]
# plot_acc([acc2, acc3, acc4], label_list, "Global_SHT1")
# plot_acc_extr([acc2, acc3, acc4], label_list, "Global_SHT1")
# plot_acc_extr1([acc6, acc7, acc5], label_list1, "Global_SHT2")
# plot_acc_extr2([acc4, acc5], label_list2, "Global_SHT3")

#model_comp(acc4, acc5, ntu120_class, "MagPhase_RealImag")
model_comp(acc1, acc5, ntu120_class, "Baseline_RealImag")
model_comp(acc1, acc4, ntu120_class, "Baseline_MagPhase")

# for i in ["A7. throw.", "A10. clapping.","A11. reading.","A12. writing.","A13. tear up paper.",'A17. take off a shoe.', 'A20. put on a hat/cap.', "A24. kicking something.","A25. reach into pocket."]:
#     print(i)
#     for j in results:
#         print(j[i])

# for i in ['A71. make ok sign.','A72. make victory sign.','A73. staple book.', 'A74. counting money.' ,'A75. cutting nails.','A76. cutting paper (using scissors).', 'A77. snapping fingers.']:
#     print(i)
#     for j in results:
#         print(j[i])


# for i in ["A30. typing on a keyboard.","A31. pointing to something with finger.","A32. taking a selfie.","A33. check time (from watch).",'A34. rub two hands together.']:
#     print(i)
#     for j in results:
#         print(j[i])



# for i in ["A36. shake head.","A37. wipe face.","A38. salute.","A39. put the palms together.","A40. cross hands in front (say stop).","A41. sneeze/cough.","A42. staggering.","A43. falling."]:
#     print(i)
#     for j in results:
#         print(j[i])
