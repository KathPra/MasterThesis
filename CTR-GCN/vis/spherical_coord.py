from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# load data
folder ="/ceph/lprasse/MasterThesis/CTR-GCN/work_dir/ntu120/"
file="csub/azimuth/epoch1_test_each_class_acc.csv"
experiments = genfromtxt(folder+file, delimiter=',')
print(experiments.shape)

# load class dict
ntu120_class = np.loadtxt("/ceph/lprasse/MasterThesis/CTR-GCN/vis/ntu120_classes.txt", delimiter='\t',dtype=str)
class_ind = np.arange(0,120,1)
class_dict = dict(zip(class_ind, ntu120_class))
print(class_dict)

accuracy = experiments[0] # for each class
confusion = experiments[1:] # for each class i: true label (row), j: prediction (column)

a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
plt.close()

# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()