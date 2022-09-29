#%%

import imp
from turtle import pos
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import os 


#%%  initialization files
path_to_simu = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/image/"
path_to_point = glob.glob(path_to_simu + "*.pc")[0]
path_to_frame = glob.glob(path_to_simu + "*.fm")[0]
path_to_feats = glob.glob(path_to_simu + "*.txt")
path_to_feats = sorted(path_to_feats, key=lambda name: int(name[len(path_to_simu): len(name) - 4]))
# print(path_to_feats)
# path_to_feats = path_to_feats.sort()


#%%  initialization frames and mappoints
Slam = StereoSlam()
Slam.m_map.readMapFile(path_to_point)
# print(path_to_feats)
Slam.readFrameFile(path_to_frame, path_to_feats)

# %% try reprojection to validate data
fx = 1.9604215879672799e+03
fy = 1.9604215879672799e+03
cx = 9.4749198913574218e+02
cy = 4.5081295013427734e+02
cam = Camera(fx, fy, cx, cy)

frames = Slam.m_frames
for frame in frames:
    features = frame.m_features
    framePos = frame.m_pos
    Rec = frame.m_rota
    for feat in features:
        mappointPos = feat.m_mappoint.m_pos
        featPos = feat.m_pos

        point_cam = mappointPos - framePos

        pos_cam = np.matmul(Rec, point_cam)
        pos_uv = cam.project(pos_cam)
        test  = cam.lift(pos_uv)
        # print(test - pos_cam / pos_cam[2, 0])
        # print(Rec)
        print(featPos - cam.project(pos_cam))

# %% set data to kalman filter
PhiPose, QPose, QPoint, PosStd, AttStd, PointStd = np.identity(6),np.identity(6), 0.01, 0.01, 0.01, 0.01

Slam.m_estimator.m_PhiPose = PhiPose
Slam.m_estimator.m_QPose = QPose
Slam.m_estimator.m_QPoint = QPoint
Slam.m_estimator.m_PosStd = PosStd
Slam.m_estimator.m_AttStd = AttStd
Slam.m_estimator.m_PointStd = PointStd


# %%
Slam.runVIO(0)