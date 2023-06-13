# import libraries
import glob
from operator import attrgetter
import numpy as np
import matplotlib.pyplot as plt
from LeastSquare import CLS
from camera import Camera
from vcommon import *
from filter import *
from coors import *
import plotmain
import os
np.seterr(all="raise")
from scipy.spatial.transform import Rotation as R


# SLAM class
class StereoSlam:
    def __init__(self):
        self.m_frames = []
        self.m_map = Map()
        self.m_filter = KalmanFilter()
        self.m_estimator = CLS()
        self.m_camera = None

    def readFrameFile(self, path_frame, path_feats):
        i_frame = 0
        with open(path_frame, "r") as f:
            while True:
                line = f.readline()

                if line:
                    frame = self.__parseFrameLine(line)
                    if i_frame >= len(path_feats):
                        break
                    features, sow = self.__parseFeatureFile(path_feats[i_frame])
                    frame.m_time = sow
                    for feat in features:
                        feat.m_frame = frame
                    frame.m_features = features
                    self.m_frames.append(frame)
                    i_frame += 1
                else:
                    break
        
    def readFeatureFile(self, path_features):
        for path_feat in path_features:
            frame = Frame()
            # time = os.path.basename(path_feat)
            # time = time[: len(time) - 4]

            # frame.m_id = time
            # frame.m_time = time

            with open(path_feat, "r") as f:
                features, sow = self.__parseFeatureFileKitti(path_feat)
                frame.m_features = features
                frame.m_time = sow
                frame.m_id = sow
                for feat in features:
                    feat.m_frame = frame
            self.m_frames.append(frame)


    def plot(self):
        path, points = [], []

        Rec0, tec0 = self.m_frames[0].m_rota, self.m_frames[0].m_pos 
        for i in range(len(self.m_frames)):
            if i == 0:
                continue
            Rec, tec = self.m_frames[i].m_rota, self.m_frames[i].m_pos

            pos = Rec0 @ (tec - tec0)
            a = pos[1, 0]
            pos[1, 0] = pos[2, 0]
            pos[2, 0] = a            # tec.reshape(3, -1)
            path.append(pos)
            # print(tec)
        
        for id, point in self.m_map.m_points.items():
            pos = point.m_pos - tec0
            pos = (Rec0 @ (pos))
            # print(pos)
            a = pos[1, 0]
            pos[1, 0] = pos[2, 0]
            pos[2, 0] = a            # tec.reshape(3, -1)
            # pos.reshape(3, -1)

            # if np.abs(pos[0, 0]) > 50 or np.abs(pos[1, 0]) > 50 or np.abs(pos[2, 0]) > 50:
            #     continue
            points.append(pos)

        
        
        path = np.array(path)
        points = np.array(points)

        # y = path[:, 1]
        # path[:, 1] = path[:, 2]
        # path[:, 2] = y

        # y = points[:, 1]
        # points[:, 1] = points[:, 2]
        # points[:, 2] = y
        plotmain.plotPointsWithTraj(path, points, "/home/xuzhuo/Documents/code/python/01-master/visual_simulation/data/traj.svg")

        
            

    
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
                    items = line.split()
                    id, week, sow = int(items[0]), float(items[1]), float(items[2])
                    i_line += 1
                    continue
                if line:
                    features.append(self.__parseFeatureLine(line))
                else: 
                    break
        return features, sow

    def __parseFeatureFileKitti(self, feature_file):
        i_line = 0
        features = []
        with open(feature_file) as f:
            while True:
                line = f.readline()
                if i_line == 0:
                    items = line.split()
                    id, sow = int(items[0]), float(items[1])
                    i_line += 1
                    continue
                if line:
                    features.append(self.__parseFeatureLine(line))
                else: 
                    break
        return features, sow


    def __parseFeatureLine(self, line = str()):
        items = line.split()

        MappointId, u, v, du = int(items[0]), float(items[1]), float(items[2]), float(items[3])
        feature = Feature(np.array([[u], [v], [du]]), du, MappointId)

        if MappointId not in self.m_map.m_points.keys():
            # mappoint = MapPoint()
            # mappoint.m_id = MappointId
            # mappoint.m_obs.append(feature)
            # self.m_map.m_points[MappointId] = mappoint
            # feature.m_mappoint = mappoint
            return feature
        
        mappoint = self.m_map.m_points[MappointId]
        feature.m_mappoint = mappoint
        mappoint.m_obs.append(feature)
        return feature

    def runVIO(self, mode = 0, path_to_output = "./", frames_gt=[], frames_linear=[], maxtime=-1, bNoiseData = False, iteration = 1, windowsize=20):
        """Run VIO for an epoch

        Args:
            mode (int, optional): 0: Linearize observation model without liearization error. Defaults to 0.
        """
        if mode == 0:
            self.runVIOWithoutError(path_to_output, maxtime)
        elif mode == 1:
            return self.runVIOWithoutError_CLS(path_to_output, frames_gt, maxtime, bNoiseData, iteration)
        elif mode == 2:
            return self.runVIOWithoutError_FilterAllState(path_to_output, frames_gt, frames_linear, maxtime, bNoiseData, iteration)
        elif mode == 3:
            return self.runVIOWithoutError_CLS_Sequential(path_to_output, frames_gt, maxtime, bNoiseData, iteration)
        elif mode == 4:
            return self.runVIOWithoutError_FilterAllState_Window(path_to_output, frames_gt, maxtime, bNoiseData)
        elif mode == 5:
            return self.runVIOWithoutError_CLS_SW(path_to_output, frames_gt, windowsize, maxtime, bNoiseData, iteration)
        elif mode == 6:
            return self.runVIOWithoutError_Filter_SW(path_to_output, frames_gt, windowsize, maxtime, bNoiseData, iteration)
        elif mode == 7:
            return self.runVIOWithoutError_CLS_SW_Marg(path_to_output, frames_gt, windowsize, maxtime, bNoiseData, iteration)
        elif mode == 8:
            return self.runVIOWithoutError_Filter_SW_Marg(path_to_output, frames_gt, windowsize, maxtime, bNoiseData, iteration)

    def runVIOWithoutError(self, path_to_output, maxtime=-1, bNoiseData = False):
        """Run VIO without linearization error
        """
        firstTec, firstRec = 0, 0
        count = 0

        f = open(path_to_output + "."+str(maxtime)+"s."+"filter", "w")
        f.close()
        LastTime = maxtime

        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        with open(path_to_output + "."+str(maxtime)+"s."+"filter", "a") as f:
            for frame in self.m_frames:      
                # print( )   
                if frame.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frame.m_pos.copy()
                    firstRec = frame.m_rota.copy()
                    count += 1
                
                tec = frame.m_pos.copy()
                Rec = frame.m_rota.copy()
                features = frame.m_features
                if len(features) <= 10:
                    print("{0}: {1} features".format(frame.m_time, len(features)))
                self.m_filter.filter(tec, Rec, features, self.m_camera)
                posError = frame.m_rota @ (tec - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(tec)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ Rec @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180

                    
                    attError = att - att_gt
                    print(att, att_gt)
                # print(att, att_gt)
                position = firstRec @ (tec - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))


    def runVIOWithoutError_CLS(self, path_to_output, frames_gt, maxtime=-1, bNoiseData = False, iteration = 1):
        """Run VIO without liearization error and use Common Least Square method

        Args:
            path_to_output (str): path to result
        """
        frames_estimate = self.m_estimator.solveAll(self.m_frames.copy(), self.m_camera, frames_gt, maxtime, iteration)

        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.CLS.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.CLS"
        print("save to " + output)

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))

    def runVIOWithoutError_FilterAllState(self, path_to_output, frames_gt, frames_linear, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_filter.filter_AllState(self.m_frames, self.m_camera, frames_gt, frames_linear, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.FilterAllState.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.FilterAllState"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))

    def runVIOWithoutError_CLS_Sequential(self, path_to_output, frames_gt, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_estimator.solveSequential(self.m_frames, self.m_camera, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.CLS_Seq.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.CLS_Seq"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))

    def runVIOWithoutError_FilterAllState_Window(self, path_to_output, frames_gt, maxtime=-1, bNoiseData = False):
        frames_estimate = self.m_filter.filter_AllState_Window(self.m_frames, self.m_camera, frames_gt, maxtime)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.FilterAllState_Window.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.FilterAllState_Window"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))


    def runVIOWithoutError_CLS_SW(self, path_to_output, frames_gt, windowsize = 20, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_estimator.solveSW(self.m_frames, frames_gt, self.m_camera, windowsize, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.CLS_SW.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.CLS_SW"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))

    def runVIOWithoutError_CLS_SW_Marg(self, path_to_output, frames_gt, windowsize = 20, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_estimator.solveSW_Marg(self.m_frames, frames_gt, self.m_camera, windowsize, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.CLS_SW_Marg.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.CLS_SW_Marg"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))

    def runVIOWithoutError_Filter_SW(self, path_to_output, frames_gt, windowsize = 20, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_filter.solveSW(self.m_frames, frames_gt, self.m_camera, windowsize, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.Filter_SW.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.Filter_SW"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))


    def runVIOWithoutError_Filter_SW_Marg(self, path_to_output, frames_gt, windowsize = 20, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_filter.solveSW_Marg(self.m_frames, frames_gt, self.m_camera, windowsize, maxtime, iteration)
        LastTime = maxtime
        if maxtime > self.m_frames[len(self.m_frames) - 1].m_time or maxtime == -1:
            LastTime = self.m_frames[len(self.m_frames) - 1].m_time

        if bNoiseData:
            output = path_to_output + "."+str(maxtime)+"s.Filter_SW_Marg.Noise"
        else:
            output = path_to_output + "."+str(maxtime)+"s.Filter_SW_Marg"

        f = open(output, "w")
        f.close()

        count, frame_i = 0, 0
        count, frame_i = 0, 0
        with open(output, "a") as f:
            for frame_estimate in frames_estimate:
                if frame_estimate.m_time > LastTime:
                    break
                if count == 0:
                    firstTec = frames_gt[0].m_pos.copy()
                    firstRec = frames_gt[0].m_rota.copy()
                    count += 1
                frame = frames_gt[frame_i]
                posError = frame.m_rota @ (frame_estimate.m_pos - frame.m_pos)

                Rcb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).transpose()
                BLH = XYZ2BLH(frame_estimate.m_pos)
                BLH[:2] *= D2R
                Rne = BLH2NEU(BLH)
                Rnc = Rcb @ frame_estimate.m_rota @ Rne
                att = rot2att(Rnc) * R2D

                BLH_gt = XYZ2BLH(frame.m_pos)
                BLH_gt[:2] *= D2R
                Rne_gt = BLH2NEU(BLH_gt)
                Rnc_gt = Rcb @ frame.m_rota @ Rne_gt
                att_gt = rot2att(Rnc_gt) * R2D

                attError = att - att_gt
                if math.fabs(attError[0]) > 50:
                    print(att, att_gt)

                    if att_gt[0] > 150:
                        att_gt[0] -= 180
                    if att_gt[0] < -150:
                        att_gt[0] += 180

                    if att[0] > 150:
                        att[0] -= 180
                    if att[0] < -150:
                        att[0] += 180
                    attError = att - att_gt
                    print(att, att_gt)

                position = firstRec @ (frame_estimate.m_pos - firstTec)
                gt_position = firstRec @ (frame.m_pos - firstTec)
                frame_i += 1
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(frame.m_time, posError[0, 0], posError[1, 0], posError[2, 0], attError[0], attError[1], attError[2], position[0, 0], position[1, 0], position[2, 0], gt_position[0, 0], gt_position[1, 0], gt_position[2, 0]))


    def runKittiVIO_FilterMarg(self, path_to_output, path_gt, windowsize=20, iteration=1):
        self.m_map.m_points.clear()

        FrameNumInWindow = 0
        for frame in self.m_frames:
            # 1. find correspondences
            if FrameNumInWindow < windowsize:
                self.m_map.addNewFrame(frame)
                FrameNumInWindow += 1
                self.TrackLastFrame()
                continue
            self.m_map.check()


    def runKittiVIO_CLSMarg(self, path_to_output, path_gt, windowsize=20, iteration=1):
        self.m_map.m_points.clear()

        FrameNumInWindow = 0
        for frame in self.m_frames:
            # 1. find correspondences
            if FrameNumInWindow < windowsize:
                self.m_map.addNewFrame(frame)
                FrameNumInWindow += 1
                if (self.TrackLastFrame() == False):
                    #TODO: remove the latest frame and its observations
                    pass
                # continue
            self.m_estimator.solveKitti(self.m_map, self.m_camera, windowsize)
    
    def removeLatestFrame(self):
        pass

    def TrackLastFrame(self):
        """solve initial value for the latest frame
        """

        FrameNum = len(self.m_map.m_frames)
        if FrameNum <= 1:
            return
        nframe = self.m_map.m_frames[FrameNum - 1]
        usedLandmarks = self.findCorrespond(nframe)

        # construct information matrix, residual, and weight matrix
        # since here we just obtain intial value, both SWF and SWO
        # use Least Square directly
        ParamNum = 6
        fx, fy, intrin_b = self.m_camera.m_fx, self.m_camera.m_fy, self.m_camera.m_b
        print(len(usedLandmarks), "tracked")
        if len(usedLandmarks) < 2:
            print("used", len(usedLandmarks), "landmarks")
            return False
        for iter in range(3):
            N, b = np.zeros((ParamNum, ParamNum)), np.zeros((ParamNum, 1))
            pRwc, pPwc = nframe.m_rota, nframe.m_pos
            for i in range(len(nframe.m_features)):
                pfeat = nframe.m_features[i]
                if pfeat.m_mappoint not in usedLandmarks:
                    continue
                PointPos = pfeat.m_mappoint.m_pos
                PointPos_c = pRwc @ (PointPos - pPwc)
                pfeat.m_PosInCamera = PointPos_c.copy()
                uv = self.m_camera.project(PointPos_c)

                # PointPos_c = pfeat.m_PosInCamera
                J, P, L = np.zeros((3, ParamNum)), np.identity(3), np.zeros((3, 1))
                
                J_cam = Jacobian_rcam(pRwc, PointPos_c, fx, fy, intrin_b)
                Jphi = Jacobian_phai(pRwc, pPwc, PointPos, PointPos_c, fx, fy, intrin_b)

                J[: 3, 0: 3] = J_cam
                J[:3, 3: 6] = Jphi

                P = P * (1 / (self.m_filter.m_PixelStd ** 2))
                uv_obs = pfeat.m_pos
                L[:] = uv - uv_obs

                N += J.transpose() @ P @ J
                b += J.transpose() @ P @ L
            # np.savetxt("./debug/N.txt", N)
            dx = np.linalg.inv(N) @ b
            nframe.m_pos = nframe.m_pos - dx[0: 3, :]
            nframe.m_rota = nframe.m_rota @ (np.identity(3) - SkewSymmetricMatrix(dx[3: 6, :]))
        self.ProjectLandmarks(nframe)
        return True
    
    def removeOutlier(self, matchedframe, matchedFeature, iteration):
        """_summary_

        Args:
            matchedFrame (_type_): _description_
            matchedFeature (_type_): _description_
            iteration (_type_): _description_

        Returns:
            int: number of removed landmarks
        """
        threshold = 10
        if iteration >= 2:
            threshold = 3        
        
        pfeatures, nfeatures = matchedFeature[0], matchedFeature[1]
        pframe, nframe = matchedframe[0], matchedframe[1]
        pRwc, pPwc = pframe.m_rota, pframe.m_pos
        nRwc, nPwc = nframe.m_rota, nframe.m_pos

        numRemove, featureNum = 0, len(pfeatures)
        i = 0
        while i < featureNum:
            PointPos = pfeatures[i].m_mappoint.m_pos
            pPointPos_c = pRwc @ (PointPos - pPwc)
            puv = self.m_camera.project(pPointPos_c)
            puv_obs = pfeatures[i].m_pos

            nPointPos_c = nRwc @ (PointPos - nPwc)
            nuv = self.m_camera.project(nPointPos_c)
            nuv_obs = nfeatures[i].m_pos

            nremoved, premoved = False, False
            if np.linalg.norm(puv - puv_obs, 2) > threshold:
                premoved = True
                pfeatures[i].m_buse = False
            if np.linalg.norm(nuv - nuv_obs, 2) > threshold:
                nremoved = True
                nfeatures[i].m_buse = False
            
            if (pfeatures[i].m_buse == False and nfeatures[i].m_buse == False) or pPointPos_c[2, 0] <= 0 or nPointPos_c[2, 0] <= 0:
                pfeatures[i].m_mappoint.m_buse = False
                del pfeatures[i]
                del nfeatures[i]
                numRemove += 1
                i -= 1
            featureNum = len(pfeatures)
            i += 1

        return numRemove

    def ProjectLandmarks(self, frame):
        for feat in frame.m_features:
            Rc, rc = frame.m_rota, frame.m_pos
            mappoint = feat.m_mappoint
            if np.linalg.norm(mappoint.m_pos) != 0:
                continue
            
            feat.m_PosInCamera = self.m_camera.lift(feat.m_pos)
            mappoint.m_pos = rc + np.linalg.inv(Rc) @ feat.m_PosInCamera


    # def findCorrespond(self, pframe, nframe):
    #     pairs = {}
    #     count_feat = 0
    #     count_land = 0
    #     num = 0
    #     for pfeat in pframe.m_features:
    #         mappoint = pfeat.m_mappoint
    #         if mappoint.m_buse == False:
    #             count_land += 1
    #             continue
    #         for obs in mappoint.m_obs:
    #             if obs in nframe.m_features:
    #                 num += 1
    #                 if obs.m_buse == False and pfeat.m_buse == False:
    #                     count_feat += 1
    #                     continue
    #                 pairs[pfeat] = obs
    #                 continue
    #     print(num, "pairs found, ", count_feat, "reject")
    #     return pairs

    def findCorrespond(self, nframe):
        usedLandmarks = []
        for feat in nframe.m_features:
            mappoint = feat.m_mappoint
            if mappoint.m_buse < 1:
                continue
            if np.linalg.norm(mappoint.m_pos, 2) == 0:
                continue
            usedLandmarks.append(mappoint)
        
        return usedLandmarks

