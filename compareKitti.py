import numpy as np
import plotmain as main
import glob
from vcommon import *

# define attribute
plotAttributes = {}
posAttributes = {}
posAttributes["ylabel"] = "Position Error(m)"
posAttributes["xlabel"] = "Epoch(sec)"
posAttributes["legend"] = ["R", "F", "U"]
posAttributes["xlim"] = [0, 120]
posAttributes["ylim"] = [-8, 8]
posAttributes["scientific"] = False

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
velAttributes["xlim"] = [0, 0]
velAttributes["ylim"] = [0.6, 0.6]
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
attAttributes["ylabel"] = "Attitude Error(Deg)"
attAttributes["xlabel"] = "Epoch(sec)"
attAttributes["legend"] = ["Y", "P", "R"]
attAttributes["xlim"] = [0, 120]
attAttributes["ylim"] = [-6, 6]
attAttributes["scientific"] = False


attSubAtt = {}
attSubAtt["xpos"] = 0.1
attSubAtt["ypos"] = 0.65
attSubAtt["width"] = 0.4
attSubAtt["height"] = 0.3
attSubAtt["range"] = 50, 650
attSubAtt["ylim"] = [0.5, 0.5]
attSubAtt["xlim"] = [0, 0]
attSubAtt["loc"] = [1, 2]
attAttributes["subplot"] = attSubAtt

plotAttributes['pos'] = posAttributes
plotAttributes['vel'] = velAttributes
plotAttributes['att'] = attAttributes

orders = ["pos", "att"]
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
cmp = "kitti"
names = [".CLS_Seq", ".CLS"]

#%% CLS - FILTER
CLS = np.loadtxt(prefix + "kitti_07_CLSMarg.txt")
Filter = np.loadtxt(prefix + "kitti_07_FilterMarg.txt")

def findGT(time, gt):
    pos, rota = 0, 0
    for row in range(gt.shape[0]):
        if np.abs(time - gt[row, 0]) >= 1E-1:
            continue
        Trans = np.reshape(gt[row, 1:], (3, 4))
        rota, pos = Trans[:3, :3], np.reshape(Trans[:3, 3], (3,1))

    return rota, pos

#%% absolute error (CLS)
gtprefix = "/media/xuzhuo/T7/01-data/07-kitti/07/"
gt_file = gtprefix + "07.txt"
gttimes_file = gtprefix + "times.txt"

gt = np.loadtxt(gt_file)
gttimes = np.loadtxt(gttimes_file)
gt = np.insert(gt, 0, gttimes, axis=1)

pos_error, att_error = [], []

for row in range(CLS.shape[0]):
    rota_gt, pos_gt = findGT(CLS[row, 0], gt)
    att_gt = rot2att(np.linalg.inv(rota_gt)) * R2D
    att, pos = CLS[row, 4: ], np.reshape(CLS[row, 1: 4], (3, 1))

    for i in range(3):
        if att_gt[i] > 150:
            att_gt[i] -= 180
        if att_gt[i] < -150:
            att_gt[i] += 180

        if att[i] > 150:
            att[i] -= 180
        if att[i] < -150:
            att[i] += 180

    att_error.append(att - att_gt)
    pos_error.append((pos_gt - pos).transpose().flatten())

time = CLS[:, 0]
time -= time[0]
att_error, pos_error = np.array(att_error), np.array(pos_error)
main.ploterror(time, att_error, prefix + "/" + "kitti_att_cls.svg", plotAttributes["att"], False)
main.ploterror(time, pos_error, prefix + "/" + "kitti_pos_cls.svg", plotAttributes["pos"], False)

#%% absolut 

pos_error, att_error = [], []
for row in range(Filter.shape[0]):
    rota_gt, pos_gt = findGT(Filter[row, 0], gt)
    att_gt = rot2att(np.linalg.inv(rota_gt)) * R2D
    att, pos = Filter[row, 4: ], np.reshape(Filter[row, 1: 4], (3, 1))

    for i in range(3):
        if att_gt[i] > 150:
            att_gt[i] -= 180
        if att_gt[i] < -150:
            att_gt[i] += 180

        if att[i] > 150:
            att[i] -= 180
        if att[i] < -150:
            att[i] += 180

    att_error.append(att - att_gt)
    pos_error.append((pos_gt - pos).transpose().flatten())

time = Filter[:, 0]
time -= time[0]
att_error, pos_error = np.array(att_error), np.array(pos_error)
main.ploterror(time, att_error, prefix + "/" + "kitti_att_filter.svg", plotAttributes["att"], False)
main.ploterror(time, pos_error, prefix + "/" + "kitti_pos_filter.svg", plotAttributes["pos"], False)
