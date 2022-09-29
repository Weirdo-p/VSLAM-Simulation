# import libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
from vcommon import *
from filter import *

from scipy.spatial.transform import Rotation as R


# SLAM class
class StereoSlam:
    def __init__(self):
        self.m_frames = []
        self.m_map = Map()
        self.m_estimator = KalmanFilter()

    def readFrameFile(self, path_frame, path_feats):
        i_frame = 0
        with open(path_frame, "r") as f:
            while True:
                line = f.readline()

                if line:
                    frame = self.__parseFrameLine(line)
                    if i_frame >= len(path_feats):
                        break
                    features = self.__parseFeatureFile(path_feats[i_frame])
                    for feat in features:
                        feat.m_frame = frame
                    frame.m_features = features
                    self.m_frames.append(frame)
                    i_frame += 1
                else:
                    break
    
    def __parseFrameLine(self, line = str()):
        items = line.split()
        id, x, y, z = int(items[0]), float(items[1]), float(items[2]), float(items[3])

        pos = np.array([[x], [y], [z]])
        rota = np.zeros(9)
        j = 0
        for i in range(4, len(items)):
            rota[j] = float(items[i])
            j += 1
        # rota = att2rot(euler)
        rota = rota.reshape((3, 3))
        frame = Frame()
        frame.m_id = id
        frame.m_pos = pos
        frame.m_rota = rota
        return frame

    def __parseFeatureFile(self, feature_file):
        i_line = 0
        features = []
        with open(feature_file) as f:
            while True:
                line = f.readline()
                if i_line == 0:
                    i_line += 1
                    continue
                if line:
                    features.append(self.__parseFeatureLine(line))
                else: 
                    break
        return features

    def __parseFeatureLine(self, line = str()):
        items = line.split()

        MappointId, u, v, du = int(items[0]), float(items[1]), float(items[2]), float(items[3])
        feature = Feature(np.array([[u], [v]]), du, MappointId)

        mappoint = self.m_map.m_points[MappointId]
        feature.m_mappoint = mappoint
        mappoint.m_obs.append(feature)
        return feature

    def runVIO(self, mode = 0):
        """Run VIO for an epoch

        Args:
            mode (int, optional): 0: Linearize observation model without liearization error. Defaults to 0.
        """
        if mode == 0:
            self.runVIOWithoutError()
        pass

    def runVIOWithoutError(self):
        for frame in self.m_frames:
            pass

