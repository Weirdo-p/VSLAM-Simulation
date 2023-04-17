#%%

import imp
from turtle import pos
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import os 
import plotmain as main


#%% init result file
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
name = "result.txt.-1s.Filter_SW_Marg"
path_to_output = prefix + name


# %% plot error
error = np.loadtxt(path_to_output)
time = error[:, 0] 
time -= time[0]

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Error(m)"
posAttributes["xlabel"] = "Epoch(sec)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 70]
posAttributes["ylim"] = [-6, 6]
posAttributes["scientific"] = True

posSubAtt = {}
posSubAtt["xpos"] = 0.1
posSubAtt["ypos"] = 0.65
posSubAtt["width"] = 0.4
posSubAtt["height"] = 0.3
posSubAtt["range"] = 50, 650
posSubAtt["ylim"] = [-1.5, 1.5]
posSubAtt["xlim"] = [20, 60]
posSubAtt["loc"] = [3, 4]
posAttributes["subplot"] = posSubAtt

velAttributes = {}
velAttributes["ylabel"] = "Error(m/s)"
velAttributes["xlabel"] = "Epoch(sec)"
velAttributes["legend"] = ["R", "F", "U"]
velAttributes["xlim"] = [0, 70]
velAttributes["ylim"] = [-1.5, 1.5]
velAttributes["scientific"] = True

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
attAttributes["ylim"] = [-6, 6]
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
j = 0
for i in range(1, len(orders) * 3 - 1, 3):
    start, end = i, i + 3
    main.ploterror(time, error[:, start : end], prefix + "/" + orders[j] + name + ".svg", plotAttributes[orders[j]], False)
    print(orders[j], error[-1:, start: end] / 1593.8890000000001)
    j += 1

trajs = {}
traj_vo = error[:, 7 : 10]
traj_vo[:, 1] = traj_vo[:, 2]
traj_vo[:, 2] = error[:, 9]
traj_vo[:, 1] += 2
traj_vo[:, 2] += 5

traj_gt = error[:, 10: ]
traj_gt[:, 1] = traj_gt[:, 2]
traj_gt[:, 2] = error[:, 12]

trajs["GroundTruth"] = traj_gt
trajs["VO"] = traj_vo
main.plotTraj(time, trajs, prefix + "/" + "traj_" + name + ".svg")
