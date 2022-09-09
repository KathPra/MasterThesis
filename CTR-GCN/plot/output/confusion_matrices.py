from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

def create_CM(data, save_name):
    # plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.imshow(data, origin="upper", interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Labels', fontsize=12)
    ax.xaxis.tick_top()
    plt.savefig(f"./plot/output/{save_name}")
    plt.close()

def load_output(folder, file):
    data = genfromtxt(folder+file+".csv", delimiter=",")
    acc = data[0]
    cm = data[1:]
    return acc, cm
    
## Load data
folder = "/ceph/lprasse/MasterThesis/CTR-GCN/work_dir/"
file_azimuth = "ntu120/csub/azimuth/epoch38_test_each_class_acc"
file_longitude = "ntu120/csub/long_unnorm/epoch38_test_each_class_acc"
#file_combo = "ntu120/csub/azimuth_long_unnorm/epoch38_test_each_class_acc"
file_baseline = "ntu120/csub/baseline/epoch64_test_each_class_acc"

azimuth_acc, azimuth_cm = load_output(folder, file_azimuth)
longitude_acc, longitude_cm = load_output(folder, file_longitude)
#combo_acc, combo_cm = load_output(folder, file_combo)
baseline_acc, baseline_cm = load_output(folder, file_baseline)

## Plot Confusion Matrices
create_CM(azimuth_cm, "CM_azimuth_e38")
create_CM(longitude_cm, "CM_longitude_e38")
create_CM(baseline_cm, "CM_baseline_e64")

## Plot Accuracy comparison
classes = np.arange(0,120)
plt.plot(classes, baseline_acc, color='blue', label="Baseline", linewidth=0.7)
plt.plot(classes, azimuth_acc, color='red', label="Azimuth", linewidth=0.7)
plt.plot(classes, longitude_acc, color='green', label="Longitude", linewidth=0.7)
plt.xlabel('Classes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
#plt.grid(True)
plt.legend()
plt.savefig("./plot/output/acc_comparison")
plt.close()


## Determine most different classes (top 10)
diff_azimuth = baseline_acc - azimuth_acc
azimuth_worstdiff_ind = diff_azimuth.argsort()[-10:][::-1] # [::-1] sorts from worst to best w/i top 10 worst accuracies
azimuth_worstdiff_acc = diff_azimuth[azimuth_worstdiff_ind] # print corresponding accuracy differnces
print(azimuth_worstdiff_ind,azimuth_worstdiff_acc)
# A92. move heavy objects, A1. drink water, A103. yawn, A83. ball up paper, A84. play magic cube, A12. writing,A114. carry something with other person,
# A36. shake head, A11. reading, A104. stretch oneself

diff_longitude = baseline_acc - longitude_acc
longitude_worstdiff_ind = diff_longitude.argsort()[-10:][::-1] # [::-1] sorts from worst to best w/i top 10 worst accuracies
longitude_worstdiff_acc = diff_longitude[longitude_worstdiff_ind] # print corresponding accuracy differnces
print(longitude_worstdiff_ind,longitude_worstdiff_acc)
# A92. move heavy objects, A84. play magic cube, A114. carry something with other person, A11. reading, A78. open bottle, A91. open a box, A1. drink water,
# A107. wield knife towards other person, A82. fold paper, A18. wear on glasses


## Determine worst classes for each model
baseline_worst10_ind = baseline_acc.argsort()[:10]
baseline_worst10_acc = baseline_acc[baseline_worst10_ind] # print corresponding accuracy differnces
print(baseline_worst10_ind,baseline_worst10_acc)
# A73. staple book, A72. make victory sign, A71. make ok sign, A29. playing with phone/tablet, A12. writing, A74. counting money, A76. cutting paper (using scissors),
# A75. cutting nails, A106. hit other person with something, A105. blow nose

azimuth_worst10_ind = azimuth_acc.argsort()[:10]
azimuth_worst10_acc = azimuth_acc[azimuth_worst10_ind] # print corresponding accuracy differnces
print(azimuth_worst10_ind,azimuth_worst10_acc)
# A12. writing, A73. staple book, A72. make victory sign, A71. make ok sign, A74. counting money, A103. yawn, A84. play magic cube, A11. reading,
# A76. cutting paper (using scissors), A69. thumb up.
# unique (not in longitude): A103. yawn, A76. cutting paper (using scissors), A69. thumb up
# unique (not in baseline): A11. reading, A69. thumb up, A84. play magic cube, A103. yawn

longitude_worst10_ind = longitude_acc.argsort()[:10]
longitude_worst10_acc = longitude_acc[longitude_worst10_ind] # print corresponding accuracy differnces
print(longitude_worst10_ind,longitude_worst10_acc)
# A72. make victory sign,  A73. staple book, A12. writing, A71. make ok sign,  A11. reading, A74. counting money,  A84. play magic cube, A78. open bottle,
# A107. wield knife towards other person, A106. hit other person with something
# unique (not in longitude): A78. open bottle, A106. hit other person with something, A107. wield knife towards other person
# unique (not in baseline): A11. reading, A78. open bottle, A84. play magic cube,A107. wield knife towards other person

# 
print(np.sort(azimuth_worst10_ind))

print(np.sort(longitude_worst10_ind))