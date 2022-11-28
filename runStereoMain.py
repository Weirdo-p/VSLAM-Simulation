#%%
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import sys
#%%  initialization files
sys.setrecursionlimit(3000)
path_to_simu = "/home/weirdo/Documents/code/python/VSLAM-Simulation/data/"
path_to_point = glob.glob(path_to_simu + "*.pc.noise")[0]
path_to_frame = glob.glob(path_to_simu + "*.fm.noise")[0]
path_to_feats = glob.glob(path_to_simu + "*.txt")
path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

# read ground truth file
path_to_point_gt = glob.glob(path_to_simu + "*.pc")[0]
path_to_frame_gt = glob.glob(path_to_simu + "*.fm")[0]
path_to_feats_gt = glob.glob(path_to_simu + "*.txt")
path_to_feats_gt = sorted(path_to_feats_gt, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

#%% init result file
prefix = "/home/weirdo/Documents/code/python/VSLAM-Simulation/log/"
path_to_output = prefix + "result.txt"
# f = open(path_to_output, "w")
# f.close()
# print(path_to_feats)
# path_to_feats = path_to_feats.sort()


#%%  initialization frames and mappoints

#%%
Slam_gt = StereoSlam()
Slam_gt.m_map.readMapFile(path_to_point_gt)
# print(path_to_feats)
Slam_gt.readFrameFile(path_to_frame_gt, path_to_feats_gt)

# %%
time, step = -1,  0.2
while time <= -1:
    Slam = StereoSlam()
    Slam.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam.readFrameFile(path_to_frame_gt, path_to_feats_gt)

    # %% try reprojection to validate data and init camera
    fx = 1.9604215879672799e+03
    fy = 1.9604215879672799e+03
    cx = 9.4749198913574218e+02
    cy = 4.5081295013427734e+02
    b = 801.8527356
    cam = Camera(fx, fy, cx, cy, b)

    # %% set data to kalman filter
    PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0, 0, 10, 10 * D2R, 10, 2

    Slam.m_filter.m_PhiPose = PhiPose
    Slam.m_filter.m_QPose = QPose
    Slam.m_filter.m_QPoint = QPoint
    Slam.m_filter.m_PosStd = PosStd
    Slam.m_filter.m_AttStd = AttStd
    Slam.m_filter.m_PointStd = PointStd
    Slam.m_filter.m_PixelStd = PixelStd

    Slam.m_estimator.m_PixelStd = PixelStd
    Slam.m_estimator.m_PosStd = PosStd
    Slam.m_estimator.m_AttStd = AttStd
    Slam.m_estimator.m_PointStd = PointStd
    Slam.m_camera = cam
    Slam.runVIO(5, path_to_output, Slam_gt.m_frames, time, False, 1, 20)
    time += step
