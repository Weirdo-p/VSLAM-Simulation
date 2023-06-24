# %% 
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import sys

path_to_kitti = "/media/xuzhuo/T7/01-data/07-kitti/07/KITTI_07/Features/"
path_to_feats = glob.glob(path_to_kitti + "*.txt")
path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_kitti): len(name) - 4]))

result_prefix = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/"
path_to_output = result_prefix + "result.txt"


kitti_slam = StereoSlam()
kitti_slam.readFeatureFileXyz(path_to_feats)

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

kitti_slam.runKittiVIO_CLSMarg(path_to_output, "./")
# %%
