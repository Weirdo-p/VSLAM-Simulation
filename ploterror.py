from turtle import pos
import numpy as np
import coors
import math
import main
import matplotlib.pyplot as plt

prefix = "/home/xuzhuo/Documents/code/python/plot/data_20211115/28/"
err_path = prefix + "NMPL19430024P_57.flf.imu_VIO.cmp"

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Error(m)"
posAttributes["xlabel"] = "Epoch(sec)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 300]
posAttributes["ylim"] = [-8, 8]
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
attAttributes["xlim"] = [0, 300]
attAttributes["ylim"] = [-0.8, 0.8]

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

error = np.loadtxt(err_path, dtype=np.float32, comments="#")

time = error[:, 0]
time -= time[0]

orders = ["pos", "vel", "att"]

j = 0
for i in range(1, 8, 3):
    start, end = i, i + 3
    main.ploterror(time, error[:, start : end], prefix + "/" + orders[j] + ".svg", plotAttributes[orders[j]], False)
    print(orders[j], error[-1:, start: end])
    j += 1


