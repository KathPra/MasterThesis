
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-1,1,0.1)
y_cos = np.cos(x)
y_acos = np.arccos(x)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for g in np.unique(labels):
#     j = np.where(labels == g)
#     ax.scatter(x_val[j], y_val[j], z_val[j], label=label_dict[g]) 
plt.plot(x,y_cos, label="Cosine")
plt.plot(x, y_acos, label="Cosine^-1")
# plt.set_xlim(-1,1)
# plt.set_ylim(-1,1)
# plt.set_xticks([-1,-0.5,0,0.5,1])
# plt.set_yticks([-1,-0.5,0,0.5,1])
# plt.set_xlabel("x")
# plt.set_ylabel("y")
plt.legend()
plt.savefig("cos.png")
plt.close()