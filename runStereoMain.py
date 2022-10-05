#%%

import imp
from turtle import pos
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import os 
import main


#%%  initialization files
path_to_simu = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/image/"
path_to_point = glob.glob(path_to_simu + "*.pc")[0]
path_to_frame = glob.glob(path_to_simu + "*.fm")[0]
path_to_feats = glob.glob(path_to_simu + "*.txt")
path_to_feats = sorted(path_to_feats, key=lambda name: int(name[len(path_to_simu): len(name) - 4]))

#%% init result file
prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
path_to_output = prefix + "result.txt"
f = open(path_to_output, "w")
f.close()
# print(path_to_feats)
# path_to_feats = path_to_feats.sort()


#%%  initialization frames and mappoints
Slam = StereoSlam()
Slam.m_map.readMapFile(path_to_point)
# print(path_to_feats)
Slam.readFrameFile(path_to_frame, path_to_feats)

# %% try reprojection to validate data and init camera
fx = 1.9604215879672799e+03
fy = 1.9604215879672799e+03
cx = 9.4749198913574218e+02
cy = 4.5081295013427734e+02
b = 801.8527356
cam = Camera(fx, fy, cx, cy, b)

# frames = Slam.m_frames
# for frame in frames:
#     features = frame.m_features
#     framePos = frame.m_pos
#     Rec = frame.m_rota
#     for feat in features:
#         mappointPos = feat.m_mappoint.m_pos
#         featPos = feat.m_pos

#         point_cam = mappointPos - framePos

#         pos_cam = np.matmul(Rec, point_cam)
#         pos_uv = cam.project(pos_cam)
#         test  = cam.lift(pos_uv)
#         # print(test - pos_cam / pos_cam[2, 0])
#         # print(Rec)
#         print(featPos)
#         print(cam.project(pos_cam))
#         print(featPos - cam.project(pos_cam))

# %% set data to kalman filter
PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0.1, 0.0, 0.01, 0.01, 0.001, 2

Slam.m_estimator.m_PhiPose = PhiPose
Slam.m_estimator.m_QPose = QPose
Slam.m_estimator.m_QPoint = QPoint
Slam.m_estimator.m_PosStd = PosStd
Slam.m_estimator.m_AttStd = AttStd
Slam.m_estimator.m_PointStd = PointStd
Slam.m_estimator.m_PixelStd = PixelStd
Slam.m_camera = cam


# %%
Slam.runVIO(0, path_to_output)


# # %% plot error
# error = np.loadtxt(path_to_output)
# time = error[:, 0] / 60.0
# time -= time[0]

# # define attribute
# plotAttributes = {}
# posAttributes = {}
# posAttributes["ylabel"] = "Error(m)"
# posAttributes["xlabel"] = "Epoch(min)"
# posAttributes["legend"] = ["R", "F", "U"]
# posAttributes["xlim"] = [0, 25]
# posAttributes["ylim"] = [-0.4, 0.4]
# posSubAtt = {}
# posSubAtt["bplot"] = False
# posSubAtt["xpos"] = 0.25
# posSubAtt["ypos"] = 0.08
# posSubAtt["width"] = 0.4
# posSubAtt["height"] = 0.3
# posSubAtt["range"] = 4000, 6500
# posSubAtt["ylim"] = [-0.1, 0.1]
# posSubAtt["xlim"] = [450, 650]
# posSubAtt["loc"] = [3, 4]
# posAttributes["subplot"] = posSubAtt

# attAttributes = {}
# attAttributes["ylabel"] = "Error(Deg)"
# attAttributes["xlabel"] = "Epoch(min)"
# attAttributes["legend"] = ["Y", "P", "R"]
# attAttributes["xlim"] = [0, 25]
# attAttributes["ylim"] = [-1, 1]

# attSubAtt = {}
# attSubAtt["bplot"] = True
# attSubAtt["xpos"] = 0.5
# attSubAtt["ypos"] = 0.1
# attSubAtt["width"] = 0.35
# attSubAtt["height"] = 0.25
# attSubAtt["range"] = 12 * 600, 14 * 600
# attSubAtt["ylim"] = [-0.0000005, 0.0000005]
# attSubAtt["xlim"] = [12, 14]
# attSubAtt["loc"] = [1, 2]
# attAttributes["subplot"] = attSubAtt

# plotAttributes['pos'] = posAttributes
# plotAttributes['att'] = attAttributes

# orders = ["pos", "att"]
# j = 0
# for i in range(1, len(orders) * 3 - 1, 3):
#     start, end = i, i + 3
#     main.ploterror(time, error[:, start : end], prefix + "/" + orders[j] + ".svg", plotAttributes[orders[j]])
#     print(orders[j], error[-1:, start: end] / 1593.8890000000001)
#     j += 1


# %%
