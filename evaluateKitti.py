'''
this script is used for evaluate Monte Carlo experiments
'''
#%%
import os
import numpy as np
from plotmain import *

def evaluateSWMarg(subfolders, sw_file_name):
    rms_pos, rms_att = [], []

    for subfolder in subfolders:
        result_file = datapath + subfolder + "/" + sw_file_name
        results = np.loadtxt(result_file)

        rms_list_pos, rms_list_att = [], []
        # pos = results[:, 1]
        # print(RMS(np.zeros(pos.shape), pos))
        # print(np.linalg.norm(pos, 2, 1).shape)
        # RMS at each epoch
        for i in range(results.shape[0]):
            pos_data, att_data =  results[0: i + 1, 1: 4], results[0: i + 1, 4: 7]
            # pos_data, att_data =  np.linalg.norm(results[0: i + 1, 1: 4], 2, 1), np.linalg.norm(results[0: i + 1, 4: 7], 2, 1)
            rms_list_pos.append(RMS(np.zeros(pos_data.shape), pos_data))
            rms_list_att.append(RMS(np.zeros(att_data.shape), att_data))
            # rms_list_pos.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 1: 4].reshape((i + 1) * 3, -1)))
            # rms_list_att.append(RMS(np.zeros(((i + 1) * 3, 1)), results[0: i + 1, 4: 7].reshape((i + 1) * 3, -1)))
        
        rms_pos.append(np.array(rms_list_pos))
        rms_att.append(np.array(rms_list_att))
    # print (rms_pos)
    mean_rms_pos = np.mean(np.array(rms_pos).transpose(), 1)
    mean_rms_att = np.mean(np.array(rms_att).transpose(), 1)
    # print(mean_rms_pos)
    return results[:, 0], mean_rms_pos, mean_rms_att

#%% initialize parameters
basepath = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/data_2/"
datapath = basepath + "data/"
subfolders = sorted(os.listdir(datapath), key=lambda x : int(x))
savepath = basepath + "results/" 

#%% plot kitti traj

def evaluateKitti(CLSPath, FilterPath, gtfile, timestamp):
    result_swo, result_mswf = np.loadtxt(CLSPath), np.loadtxt(FilterPath)
    pos_swo, att_swo = result_swo[:, 1: 4], result_swo[:, 4: 7]
    pos_mswf, att_mswf = result_mswf[:, 1: 4], result_mswf[:, 4: 7]
    error_pos, error_att = pos_swo - pos_mswf, att_swo - att_mswf
    time = result_swo[:, 0]

    gt, timestamp = np.loadtxt(gtfile), np.loadtxt(timestamp)
    gt_pos = []
    for i in range(time.shape[0]):
        row = -1
        for j in range(timestamp.shape[0]):
            # print(timestamp[j][: -4])
            # print(np.abs(time[i] - timestamp[j][: -4]))
            # break
            if np.abs(time[i] - timestamp[j]) < 1E-2:
                # print(float(timestamp[j][: -4]))
                row = j
                break
        if row == -1:
            print(time[i])
            continue 
        T_gt = gt[row, :].reshape(3, 4)
        gt_pos.append(T_gt[:, -1])

    return np.array(gt_pos), np.linalg.norm(error_pos, 2, 1), np.linalg.norm(error_att, 2, 1)

#%% KITTI 00
basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/00/"
CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
gtfile, timestampfile = basepath + "00.txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt("./log/gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 150]
# logyAttributes["ylim"] = [-50, 550, 150]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 150]
# logyAttributes["ylim"] = [-50, 550, 150]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# # %% KITTI 01
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/01/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + "01.txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt("./log/gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-100, 1900, 500]
# logyAttributes["ylim"] = [-1500, 500, 500]
# logyAttributes["scientific"] = True
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-100, 1900, 500]
# logyAttributes["ylim"] = [-1500, 500, 500]
# logyAttributes["scientific"] = True
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 02
# num = "02"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt("./log/gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 900, 300]
# logyAttributes["ylim"] = [-100, 1100, 300]
# logyAttributes["scientific"] = True
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 900, 300]
# logyAttributes["ylim"] = [-100, 1100, 300]
# logyAttributes["scientific"] = True
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# # %% KITTI 03
# num = "03"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-100, 500, 100]
# logyAttributes["ylim"] = [-200, 400, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-100, 500, 100]
# logyAttributes["ylim"] = [-200, 400, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# # %% KITTI 04
# num = "04"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-250, 250, 100]
# logyAttributes["ylim"] = [-50, 450, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-250, 250, 100]
# logyAttributes["ylim"] = [-50, 450, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 05
# num = "05"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 100]
# logyAttributes["ylim"] = [-150, 450, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 100]
# logyAttributes["ylim"] = [-150, 450, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")


# #%% KITTI 06
# num = "06"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 100]
# logyAttributes["ylim"] = [-250, 350, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-300, 300, 100]
# logyAttributes["ylim"] = [-250, 350, 100]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 07
# num = "07"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-250, 50, 60]
# logyAttributes["ylim"] = [-130, 170, 60]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-250, 50, 60]
# logyAttributes["ylim"] = [-130, 170, 60]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 08
# num = "08"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-500, 500, 250]
# logyAttributes["ylim"] = [-300, 700, 250]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-500, 500, 250]
# logyAttributes["ylim"] = [-300, 700, 250]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 09
# num = "09"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-400, 600, 200]
# logyAttributes["ylim"] = [-200, 800, 200]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-400, 600, 200]
# logyAttributes["ylim"] = [-200, 800, 200]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att.svg")

# #%% KITTI 10
# num = "10"
# basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI/" + num + "/"
# CLSfilename, filterfilename = "result.txtkitti_07_CLSMarg11.txt", "result.txtkitti_07_FilterMarg3.txt"
# gtfile, timestampfile = basepath + num + ".txt", basepath + "times.txt"
# gt_pos, error_pos, error_att = evaluateKitti(basepath + CLSfilename, basepath + filterfilename, gtfile, timestampfile)
# np.savetxt(basepath + "gt.txt", gt_pos)

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Position Error (m)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-50, 750, 200]
# logyAttributes["ylim"] = [-400, 400, 200]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_pos), logyAttributes, basepath + "traj_pos1.svg")

# logyAttributes = {}
# logyAttributes["ylabel"] = "y (m)"
# logyAttributes["xlabel"] = "x (m)"
# logyAttributes["barlabel"] = "Relative Attitude Error (deg)"
# logyAttributes["barrange"] = [1E-10, 1E-6]
# logyAttributes["barfraction"] = 0.046
# # logyAttributes["legend"] = ["Attitude"]
# logyAttributes["xlim"] = [-50, 750, 200]
# logyAttributes["ylim"] = [-400, 400, 200]
# logyAttributes["scientific"] = False
# plotTrajWithError2D(gt_pos, np.abs(error_att), logyAttributes, basepath + "traj_att1.svg")

# %% discrepancy analysis (ablation experiments)
basepath = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/KITTI/"
CLSfilename, ignore_b_FEJ, ignore_FEJ, swf_sa = "result.txtkitti_07_CLSMarg11.txt", "result_ignore_b_FEJ.txtkitti_07_FilterMarg3.txt", "result_ignore_FEJ.txtkitti_07_FilterMarg3.txt", "result.txtkitti_07_FilterMarg3.txt"

# %%
pos_mean_dict, att_mean_dict = {}, {}
pos_max_dict, att_max_dict = {}, {}
pos_min_dict, att_min_dict = {}, {}

for i in range(11):
    if i == 0:
        continue

    seq_num = "{0:02d}".format(i)
    kitti_seq = basepath + seq_num + "/"
    gtfile, timestampfile = kitti_seq + seq_num + ".txt", kitti_seq + "times.txt"
    gt_pos, error_pos_ignore_b_FEJ, error_att_ignore_b_FEJ = evaluateKitti(kitti_seq + CLSfilename, kitti_seq + ignore_b_FEJ, gtfile, timestampfile)
    gt_pos, error_pos_ignore_FEJ  , error_att_ignore_FEJ   = evaluateKitti(kitti_seq + CLSfilename, kitti_seq + ignore_FEJ, gtfile, timestampfile)
    gt_pos, error_pos_swf_sa      , error_att_swf_sa       = evaluateKitti(kitti_seq + CLSfilename, kitti_seq + swf_sa, gtfile, timestampfile)
    pos_mean = [np.mean((error_pos_ignore_b_FEJ)), np.mean((error_pos_ignore_FEJ)), np.mean((error_pos_swf_sa))]
    att_mean = [np.mean((error_att_ignore_b_FEJ)), np.mean((error_att_ignore_FEJ)), np.mean((error_att_swf_sa))]
    pos_max = [np.max((error_pos_ignore_b_FEJ)), np.max((error_pos_ignore_FEJ)), np.max((error_pos_swf_sa))]
    att_max = [np.max((error_att_ignore_b_FEJ)), np.max((error_att_ignore_FEJ)), np.max((error_att_swf_sa))]
    pos_min = [np.min((error_pos_ignore_b_FEJ)), np.min((error_pos_ignore_FEJ)), np.min((error_pos_swf_sa))]
    att_min = [np.min((error_att_ignore_b_FEJ)), np.min((error_att_ignore_FEJ)), np.min((error_att_swf_sa))]

    pos_mean_dict[seq_num] = pos_mean
    att_mean_dict[seq_num] = att_mean
    pos_max_dict[seq_num] = pos_max
    att_max_dict[seq_num] = att_max
    pos_min_dict[seq_num] = pos_min
    att_min_dict[seq_num] = att_min

for key in pos_mean_dict.keys():
    for i in range(3):
        print(pos_max_dict[key][i], pos_mean_dict[key][i], end=" ")
        print(att_max_dict[key][i], att_mean_dict[key][i], end=" ")
    # print("\n")

x_label = ["SWF-Marg", "SWF-Full", "SWF-SA"]
logyAttributes = {}
logyAttributes["ylabel"] = "y (m)"
logyAttributes["xlabel"] = "x (m)"
logyAttributes["barlabel"] = "Relative Position Error (m)"
logyAttributes["barrange"] = [1E-10, 1E-10]
logyAttributes["barfraction"] = 0.046
logyAttributes["legend"] = ["Attitude"]
logyAttributes["xlim"] = [750, 750, 200]
logyAttributes["ylim"] = [-0.3, 10]
logyAttributes["scientific"] = False

# plotLineWithDot(x_label, pos_dict, logyAttributes)
# plt.figure(0)
# plt.plot(x_label,pos_max)
# # plt.xticks(x_label)
# plt.show()