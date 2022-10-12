#%%
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np

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

# %% set data to kalman filter
PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0, 0, 3, 5 * D2R, 10, 2

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
