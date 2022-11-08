import numpy as np
import glob

prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
H_filter_files = glob.glob( prefix + "H_FILTER_*.txt")
H_CLS_files = glob.glob(prefix + "H_CLS_*.txt")

H_filter_files = sorted(H_filter_files, key=lambda name: int(name[len(prefix) + 9: len(name) - 4]))
H_CLS_files = sorted(H_CLS_files, key=lambda name: int(name[len(prefix) + 6: len(name) - 4]))
# print(H_filter_files)
# H_CLS = np.loadtxt(H_CLS_file)

H_filters, H_CLS_Vec = [], []

for i in range(len(H_filter_files)):
    H_filters.append(np.loadtxt(H_filter_files[i]))

for i in range(len(H_CLS_files)):
    H_CLS_Vec.append(np.loadtxt(H_CLS_files[i]))


for i in range(len(H_CLS_Vec)):
    np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_diff_" + str(i) + ".txt", H_filters[i] - H_CLS_Vec[i])


prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
L_filter_files = glob.glob( prefix + "L_FILTER_*.txt")
L_CLS_files = glob.glob(prefix + "L_CLS_*.txt")

L_filter_files = sorted(L_filter_files, key=lambda name: int(name[len(prefix) + 9: len(name) - 4]))
L_CLS_files = sorted(L_CLS_files, key=lambda name: int(name[len(prefix) + 6: len(name) - 4]))
# print(H_filter_files)
# H_CLS = np.loadtxt(H_CLS_file)

L_filters, L_CLS_Vec = [], []

for i in range(len(L_filter_files)):
    L_filters.append(np.loadtxt(L_filter_files[i]))

for i in range(len(L_CLS_files)):
    L_CLS_Vec.append(np.loadtxt(L_CLS_files[i]))


for i in range(len(L_CLS_Vec)):
    np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_diff_" + str(i) + ".txt", L_filters[i] - L_CLS_Vec[i])

prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
W_filter_files = glob.glob( prefix + "W_FILTER_*.txt")
W_CLS_files = glob.glob(prefix + "W_CLS_*.txt")

W_filter_files = sorted(W_filter_files, key=lambda name: int(name[len(prefix) + 9: len(name) - 4]))
W_CLS_files = sorted(W_CLS_files, key=lambda name: int(name[len(prefix) + 6: len(name) - 4]))
# print(H_filter_files)
# H_CLS = np.loadtxt(H_CLS_file)

W_filters, W_CLS_Vec = [], []

for i in range(len(W_filter_files)):
    W_filters.append(np.loadtxt(W_filter_files[i]))

for i in range(len(W_CLS_files)):
    W_CLS_Vec.append(np.loadtxt(W_CLS_files[i]))


for i in range(len(W_CLS_Vec)):
    np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/W_diff_" + str(i) + ".txt", W_filters[i] - W_CLS_Vec[i])
# H_filter = np.zeros((H_CLS.shape[0] - H_CLS.shape[1], H_CLS.shape[1]))

# pos = 0
# for i in range(len(H_filters)):
#     H_filter[pos: pos + H_filters[i].shape[0], :] = H_filters[i]
#     pos += H_filters[i].shape[0]

# H_diff = H_filter - H_CLS[H_CLS.shape[1]:, :]

# np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_diff.txt", H_diff)
# print(H_filter.shape, H_CLS.shape)

# # L matrix
# L_filter_files = glob.glob( prefix + "L_FILTER_*.txt")
# L_CLS_file = prefix + "L_CLS.txt"

# L_filter_files = sorted(L_filter_files, key=lambda name: int(name[len(prefix) + 9: len(name) - 4]))
# print(L_filter_files)
# L_CLS = np.loadtxt(L_CLS_file)
# L_CLS.reshape(L_CLS.shape[0], 1)
# L_filters = []

# for i in range(len(L_filter_files)):
#     L_filters.append(np.loadtxt(L_filter_files[i]))

# L_filter = np.zeros((L_CLS.shape[0] - H_CLS.shape[1]))

# pos = 0
# for i in range(len(L_filters)):
#     L_filter[pos: pos + L_filters[i].shape[0]] = L_filters[i]
#     pos += L_filters[i].shape[0]

# L_diff = L_filter - L_CLS[H_CLS.shape[1]:]

# np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_diff.txt", L_diff)
# print(H_filter.shape, H_CLS.shape)
