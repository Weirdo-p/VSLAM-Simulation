#%%

import imp
from turtle import pos
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import os 
import main


#%% init result file
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
path_to_output = prefix + "result.txt"


# %% plot error
error = np.loadtxt(path_to_output)
time = error[:, 0] / 60.0
time -= time[0]

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Error(m)"
posAttributes["xlabel"] = "Epoch(min)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 25]
posAttributes["ylim"] = [-0.4, 0.4]
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
attAttributes["xlabel"] = "Epoch(min)"
attAttributes["legend"] = ["Y", "P", "R"]
attAttributes["xlim"] = [0, 25]
attAttributes["ylim"] = [-1, 1]

attSubAtt = {}
attSubAtt["bplot"] = True
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
j = 0
for i in range(1, len(orders) * 3 - 1, 3):
    start, end = i, i + 3
    main.ploterror(time, error[:, start : end], prefix + "/" + orders[j] + ".svg", plotAttributes[orders[j]])
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
main.plotTraj(time, trajs, prefix + "/" + "traj" + ".svg")

# %%
