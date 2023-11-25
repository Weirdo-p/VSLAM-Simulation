# %% 
from vcommon import *
from StereoSlam import *
from camera import *
import numpy as np
import sys
# start from 7
# for i in range(1, 11):
#     if i == 3:
#         continue
#     base = "/home/simon/visual_simulation/kitti/KITTI_" + "{0:02d}/".format(i)
#     path_to_kitti = base + "Features/"
#     path_to_feats = glob.glob(path_to_kitti + "*.txt")
#     path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_kitti): len(name) - 4]))

#     result_prefix = base
#     path_to_output = result_prefix + "result.txt"
#     path_to_campose = result_prefix + "CameraTrajectory.txt"


#     kitti_slam = StereoSlam()
#     kitti_slam.readFeatureFileXyz(path_to_feats)
#     kitti_slam.readCamPose(path_to_campose)
#     # kitti_slam.plot()

#     fx = 7.188560000000e+02
#     fy = 7.188560000000e+02
#     cx = 6.071928000000e+02
#     cy = 1.852157000000e+02
#     b =  3.861448000000e+02
#     cam = Camera(fx, fy, cx, cy, b)
#     kitti_slam.m_camera = cam
#     kitti_slam.m_map.m_camera = cam
#     PhiPose, QPose, QPoint, PosStd, AttStd, PointStd, PixelStd = np.identity(6),np.identity(6) * 0, 0, 100, 100 * D2R, 10, 3

#     kitti_slam.m_filter.m_PhiPose = PhiPose
#     kitti_slam.m_filter.m_QPose = QPose
#     kitti_slam.m_filter.m_QPoint = QPoint
#     kitti_slam.m_filter.m_PosStd = PosStd
#     kitti_slam.m_filter.m_AttStd = AttStd
#     kitti_slam.m_filter.m_PointStd = PointStd
#     kitti_slam.m_filter.m_PixelStd = PixelStd
#     kitti_slam.m_estimator.m_PixelStd = PixelStd
#     kitti_slam.m_estimator.m_PosStd = PosStd
#     kitti_slam.m_estimator.m_AttStd = AttStd
#     kitti_slam.m_estimator.m_PointStd = PointStd

#     kitti_slam.runKittiVIO_CLSMarg(path_to_output, "./")
# %%
for i in range(1, 11):
    if i== 3:
        continue
    # base = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/kitti1300/KITTI_" + "{0:02d}/".format(i)
    base = "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/kitti/03KITTI/"
    path_to_kitti = base + "Features/"
    path_to_feats = glob.glob(path_to_kitti + "*.txt")
    path_to_feats = sorted(path_to_feats, key=lambda name: float(name[len(path_to_kitti): len(name) - 4]))

    result_prefix = base
    path_to_output = result_prefix + "result.txt"
    path_to_campose = result_prefix + "CameraTrajectory.txt"


    kitti_slam = StereoSlam()
    kitti_slam.readFeatureFileXyz(path_to_feats)
    kitti_slam.readCamPose(path_to_campose)
    kitti_slam.removeRepeatedOutlier()

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