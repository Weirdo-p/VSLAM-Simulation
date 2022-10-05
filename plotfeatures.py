import numpy as np
import matplotlib.pyplot as plt


path = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/features.txt"
features = np.loadtxt(path, dtype=np.float32, comments="#")

plt.figure(figsize=(5.0393701, 3.4645669))
plt.plot(features[:, 0] - features[0, 0], features[:, 3])
plt.show()