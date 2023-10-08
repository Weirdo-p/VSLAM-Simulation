#%%
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import sys

from numpy import seterr
seterr(all='raise')
#%%  SWO simulation
PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0, 0, 10, 10 * D2R, 10, 2
sys.setrecursionlimit(3000)
batch_path = "/home/xuzhuo/Documents/code/matlab/01-Simulation_Visual_IMU/Simulation_Visual_IMU/Matlab-PSINS-PVAIMUSimulator/data/"
NumBatch = 51
for iter in range(1, NumBatch + 1):
    subfolder = batch_path + str(iter)
    path_to_simu = subfolder + "/Features/"
    path_to_output = subfolder + "/result.txt"

    path_to_point = glob.glob(path_to_simu + "*.pc.noise")[0]
    path_to_frame = glob.glob(path_to_simu + "*.fm.noise")[0]
    path_to_feats = glob.glob(path_to_simu + "*.txt")
    path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

    # read ground truth file
    path_to_point_gt = glob.glob(path_to_simu + "*.pc")[0]
    path_to_frame_gt = glob.glob(path_to_simu + "*.fm")[0]
    path_to_feats_gt = glob.glob(path_to_simu + "*.txt")
    path_to_feats_gt = sorted(path_to_feats_gt, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

    Slam_gt = StereoSlam()
    Slam_gt.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam_gt.readFrameFile(path_to_frame_gt, path_to_feats_gt)
    # Slam_gt.plot()

    Slam_linear = StereoSlam()
    Slam_linear.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam_linear.readFrameFile(path_to_frame_gt, path_to_feats_gt)

    Slam = StereoSlam()
    Slam.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam.readFrameFile(path_to_frame_gt, path_to_feats_gt)

    # init camera
    fx = 1.9604215879672799e+03
    fy = 1.9604215879672799e+03
    cx = 9.4749198913574218e+02
    cy = 4.5081295013427734e+02
    b = 801.8527356
    cam = Camera(fx, fy, cx, cy, b)

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
    Slam.runVIO(7, path_to_output, Slam_gt.m_frames, Slam_linear.m_frames, -1, True, 1, 20)

#%% SWF simulation
for iter in range(1, NumBatch + 1):
    subfolder = batch_path + str(iter)
    path_to_simu = subfolder + "/Features/"
    path_to_output = subfolder + "/result.txt"

    path_to_point = glob.glob(path_to_simu + "*.pc.noise")[0]
    path_to_frame = glob.glob(path_to_simu + "*.fm.noise")[0]
    path_to_feats = glob.glob(path_to_simu + "*.txt")
    path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

    # read ground truth file
    path_to_point_gt = glob.glob(path_to_simu + "*.pc")[0]
    path_to_frame_gt = glob.glob(path_to_simu + "*.fm")[0]
    path_to_feats_gt = glob.glob(path_to_simu + "*.txt")
    path_to_feats_gt = sorted(path_to_feats_gt, key=lambda name: float(name[len(path_to_simu): len(name) - 4]))

    Slam_gt = StereoSlam()
    Slam_gt.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam_gt.readFrameFile(path_to_frame_gt, path_to_feats_gt)
    # Slam_gt.plot()

    Slam_linear = StereoSlam()
    Slam_linear.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam_linear.readFrameFile(path_to_frame_gt, path_to_feats_gt)

    Slam = StereoSlam()
    Slam.m_map.readMapFile(path_to_point_gt)
    # print(path_to_feats)
    Slam.readFrameFile(path_to_frame_gt, path_to_feats_gt)

    # init camera
    fx = 1.9604215879672799e+03
    fy = 1.9604215879672799e+03
    cx = 9.4749198913574218e+02
    cy = 4.5081295013427734e+02
    b = 801.8527356
    cam = Camera(fx, fy, cx, cy, b)

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
    Slam.runVIO(8, path_to_output, Slam_gt.m_frames, Slam_linear.m_frames, -1, True, 1, 20)
