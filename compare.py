import numpy as np
import plotmain as main
import glob

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Error(m)"
posAttributes["xlabel"] = "Epoch(sec)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 70]
posAttributes["ylim"] = [-0.01, 0.01]
posAttributes["scientific"] = True

posSubAtt = {}
posSubAtt["xpos"] = 0.1
posSubAtt["ypos"] = 0.65
posSubAtt["width"] = 0.4
posSubAtt["height"] = 0.3
posSubAtt["range"] = 50, 650
posSubAtt["ylim"] = [-0.5, 0.5]
posSubAtt["xlim"] = [20, 60]
posSubAtt["loc"] = [3, 4]
posAttributes["subplot"] = posSubAtt

velAttributes = {}
velAttributes["ylabel"] = "Error(m/s)"
velAttributes["xlabel"] = "Epoch(sec)"
velAttributes["legend"] = ["R", "F", "U"]
velAttributes["xlim"] = [0, 300]
velAttributes["ylim"] = [-0.6, 0.6]
velAttributes["scientific"] = False

velSubAtt = {}
velSubAtt["xpos"] = 0.1
velSubAtt["ypos"] = 0.65
velSubAtt["width"] = 0.4
velSubAtt["height"] = 0.3
velSubAtt["range"] = 50, 650
velSubAtt["ylim"] = [-0.05, 0.05]
velSubAtt["xlim"] = [20, 26]
velSubAtt["loc"] = [3, 2]

velAttributes["subplot"] = velSubAtt


attAttributes = {}
attAttributes["ylabel"] = "Error(Deg)"
attAttributes["xlabel"] = "Epoch(sec)"
attAttributes["legend"] = ["Y", "P", "R"]
attAttributes["xlim"] = [0, 70]
attAttributes["ylim"] = [-0.01, 0.01]
attAttributes["scientific"] = True


attSubAtt = {}
attSubAtt["xpos"] = 0.1
attSubAtt["ypos"] = 0.65
attSubAtt["width"] = 0.4
attSubAtt["height"] = 0.3
attSubAtt["range"] = 50, 650
attSubAtt["ylim"] = [-0.5, 0.5]
attSubAtt["xlim"] = [20, 60]
attSubAtt["loc"] = [1, 2]
attAttributes["subplot"] = attSubAtt

plotAttributes['pos'] = posAttributes
plotAttributes['vel'] = velAttributes
plotAttributes['att'] = attAttributes

orders = ["pos", "att"]
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
cmp = "CLS_FilterAll"
names = [".CLS_Seq", ".CLS"]

# errors = {}

# SortFunc = lambda name, name1: float(name[78: len(name) - len(name1) - 1])
# for i in range(len(names)):
#     Files = glob.glob(prefix + "*" + names[i])
#     Files = sorted(Files, key=lambda name: float(name[78: len(name) - len(names[i]) - 1]))

#     error = []

#     for file in Files:
#         data = np.loadtxt(file)
#         print(data[-1:, ].shape)
#         error.append(data[-1:, ])
    
#     errors[names[i]] = np.array(error)
#     dim1, dim2 = errors[names[i]].shape[0], errors[names[i]].shape[2]
#     errors[names[i]] = errors[names[i]].reshape(dim1, dim2)

# print(errors[names[0]][:, 1:])
# compare = errors[names[0]][:, 1:] - errors[names[1]][:, 1:]
# # print(errors[names[0]].reshape(dim1, dim2))
# time = errors[names[0]][:, 0]

CLS = np.loadtxt(prefix + "result.txt.-1s.CLS_Seq")
Filter = np.loadtxt(prefix + "result.txt.-1s.FilterAllState")

time = CLS[:, 0]
time -= time[0]

error = CLS - Filter
orders = ["pos", "att"]
j = 0
for i in range(1, len(orders) * 3 - 1, 3):
    start, end = i, i + 3
    main.ploterror(time, error[:, start : end], prefix + "/" + orders[j] + cmp + ".svg", plotAttributes[orders[j]], False)
    print(orders[j], error[-1:, start: end] / 1593.8890000000001)
    j += 1