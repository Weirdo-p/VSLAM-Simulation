import numpy as np
import plotmain as main
import glob

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Error(m)"
posAttributes["xlabel"] = "Epoch(s)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 30]
posAttributes["ylim"] = [-0.1, 0.1]
posSubAtt = {}
posSubAtt["bplot"] = False
posSubAtt["xpos"] = 0.25
posSubAtt["ypos"] = 0.08
posSubAtt["width"] = 0.4
posSubAtt["height"] = 0.3
posSubAtt["range"] = 4000, 6500
posSubAtt["ylim"] = [-0.1, 0.1]
posSubAtt["xlim"] = [450, 650]
posSubAtt["loc"] = [3, 4]
posAttributes["subplot"] = posSubAtt

attAttributes = {}
attAttributes["ylabel"] = "Error(Deg)"
attAttributes["xlabel"] = "Epoch(s)"
attAttributes["legend"] = ["Y", "P", "R"]
attAttributes["xlim"] = [0, 30]
attAttributes["ylim"] = [-0.3, 0.3]

attSubAtt = {}
attSubAtt["bplot"] = False
attSubAtt["xpos"] = 0.5
attSubAtt["ypos"] = 0.1
attSubAtt["width"] = 0.35
attSubAtt["height"] = 0.25
attSubAtt["range"] = 12 * 600, 14 * 600
attSubAtt["ylim"] = [-0.0000005, 0.0000005]
attSubAtt["xlim"] = [12, 14]
attSubAtt["loc"] = [1, 2]
attAttributes["subplot"] = attSubAtt

plotAttributes['pos'] = posAttributes
plotAttributes['att'] = attAttributes

orders = ["pos", "att"]
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
cmp = "CLS_FilterAll"
names = [".FilterAllState", ".CLS"]

errors = {}

SortFunc = lambda name, name1: float(name[78: len(name) - len(name1) - 1])
for i in range(len(names)):
    Files = glob.glob(prefix + "*" + names[i])
    Files = sorted(Files, key=lambda name: float(name[78: len(name) - len(names[i]) - 1]))

    error = []

    for file in Files:
        data = np.loadtxt(file)
        # print(data[-1:, ].shape)
        error.append(data[-1:, ])
        print(file, data[-1:, 1])
    
    errors[names[i]] = np.array(error)
    dim1, dim2 = errors[names[i]].shape[0], errors[names[i]].shape[2]
    errors[names[i]] = errors[names[i]].reshape(dim1, dim2)
    break

# print(errors[names[0]][:, 1])
# print(errors[names[1]][:, 1])
# compare = errors[names[0]][:, 1:] - errors[names[1]][:, 1:]
# # print(errors[names[0]].reshape(dim1, dim2))
# time = errors[names[0]][:, 0]

# orders = ["pos", "att"]
# j = 0
# for i in range(1, len(orders) * 3 - 1, 3):
#     start, end = i, i + 3
#     main.ploterror(time, compare[:, start : end], prefix + "/" + orders[j] + cmp + ".svg", plotAttributes[orders[j]])
#     print(orders[j], compare[-1:, start: end] / 1593.8890000000001)
#     j += 1


# cov_CLS = np.loadtxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov_CLS.txt")
# cov_Filter = np.loadtxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/CovFilter.txt")

# np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov_Compare.txt", cov_CLS - cov_Filter)