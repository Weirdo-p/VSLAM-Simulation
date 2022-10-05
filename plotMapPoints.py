import imp
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("/home/xuzhuo/Documents/code/python/plot/data_20211115/27/mappoints.txt", dtype=np.float32)

plt.figure(figsize=(5.0393701, 3.4645669))
ax = plt.axes(projection='3d')

ax.scatter(data[:, 1], data[:, 2], data[:, 3])
ax.plot3D(data[:, 4], data[:, 5], data[:, 6], c='r')
plt.show()