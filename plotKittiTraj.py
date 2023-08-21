#%%

import imp
from turtle import pos
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import os 
import plotmain as main



# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Position Error(m)"
posAttributes["xlabel"] = "Epoch(sec)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 70]
posAttributes["ylim"] = [-1, 1]
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
attAttributes["ylabel"] = "Attitude Error(Deg)"
attAttributes["xlabel"] = "Epoch(sec)"
attAttributes["legend"] = ["Y", "P", "R"]
attAttributes["xlim"] = [0, 70]
attAttributes["ylim"] = [-1, 1]
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


#%% init result file
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
CLS = "kitti_07_CLSMarg.txt"
FILTER = "kitti_07_FilterMarg.txt"
name = "KittiTraj"
path_to_CLS = prefix + CLS
path_to_EFK = prefix + FILTER


# %% plot trajectory (X-Z plane)
cls = np.loadtxt(path_to_CLS)
cls[:, [2, 3]] = cls[:, [3, 2]]

time = cls[:, 0] 
time -= time[0]

ekf = np.loadtxt(path_to_EFK)
ekf[:, [2, 3]] = ekf[:, [3, 2]]
ekf[:, 2] = ekf[:, 2] + 5
ekf[:, 3] = ekf[:, 3] + 5

trajs = {}
trajs["MSWF"] = ekf[:, 1: 4]
trajs["SWO"] = cls[:, 1: 4]
main.plotTraj(time, trajs, prefix + "/" + "traj_" + name + ".svg")
