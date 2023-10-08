# %% 
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import sys

path_to_kitti = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI_00_1/Features/"
path_to_feats = glob.glob(path_to_kitti + "*.txt")
path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_kitti): len(name) - 4]))

result_prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/KITTI_00_1/"
path_to_output = result_prefix + "result.txt"
path_to_campose = result_prefix + "CameraTrajectory.txt"


kitti_slam = StereoSlam()
kitti_slam.readFeatureFileXyz(path_to_feats)
kitti_slam.readCamPose(path_to_campose)
# kitti_slam.plot()

fx = 7.188560000000e+02
fy = 7.188560000000e+02
cx = 6.071928000000e+02
cy = 1.852157000000e+02
b =  3.861448000000e+02
cam = Camera(fx, fy, cx, cy, b)
kitti_slam.m_camera = cam
kitti_slam.m_map.m_camera = cam
PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0, 0, 100, 100 * D2R, 10, 3

kitti_slam.m_filter.m_PhiPose = PhiPose
kitti_slam.m_filter.m_QPose = QPose
kitti_slam.m_filter.m_QPoint = QPoint
kitti_slam.m_filter.m_PosStd = PosStd
kitti_slam.m_filter.m_AttStd = AttStd
kitti_slam.m_filter.m_PointStd = PointStd
kitti_slam.m_filter.m_PixelStd = PixelStd
kitti_slam.m_estimator.m_PixelStd = PixelStd
kitti_slam.m_estimator.m_PosStd = PosStd
kitti_slam.m_estimator.m_AttStd = AttStd
kitti_slam.m_estimator.m_PointStd = PointStd

kitti_slam.runKittiVIO_FilterMarg(path_to_output, "./")
# %%
