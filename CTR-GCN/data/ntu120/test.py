import numpy as np

# cset = np.load('NTU120_CSet_SH.npz')
# # for k in cset.iterkeys():
# #     print(k)

# x_train = cset["x_train"]
# print(x_train.shape)

# y_train = cset["y_train"]
# print(y_train.shape)

# x_test = cset["x_test"]
# print(x_test.shape)

# y_test = cset["y_test"]
# print(y_test.shape)

# cset = np.load('NTU120_CSet.npz')
# x_train = cset["x_train"]
# print(x_train.shape)
# # should be (54468, 81, 64, 25, 2)
# y_train = cset["y_train"]
# print(y_train.shape)
# # should be (54468, 120)
# x_test = cset["x_test"]
# print(x_test.shape)
# # should be (59477, 81, 64, 25, 2)
# y_test = cset["y_test"]
# print(y_test.shape)
# # should be (59477, 120)

#### CSUB
csub = np.load('NTU120_CSub_SH.npz')
# for k in csub.iterkeys():
#     print(k)

x_train = csub["arr_0"]
print(x_train.shape)
# this should be (63026, 81, 64, 25, 2)

y_train = csub["arr_1"]
print(y_train.shape)
# this should be (63026, 120)

x_test = csub["arr_2"]
print(x_test.shape)
# this should be (50919, 81, 64, 25, 2)

y_test = csub["arr_3"]
print(y_test.shape)
# this should be (50919, 120)

csub = np.load('NTU120_CSub.npz')
x_train = csub["x_train"]
print(x_train.shape)

y_train = csub["y_train"]
print(y_train.shape)

x_test = csub["x_test"]
print(x_test.shape)

y_test = csub["y_test"]
print(y_test.shape)
