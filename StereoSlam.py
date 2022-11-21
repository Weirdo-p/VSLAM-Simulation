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

    def __parseFeatureLine(self, line = str()):
        items = line.split()

        MappointId, u, v, du = int(items[0]), float(items[1]), float(items[2]), float(items[3])
        feature = Feature(np.array([[u], [v], [du]]), du, MappointId)

        mappoint = self.m_map.m_points[MappointId]
        feature.m_mappoint = mappoint
        mappoint.m_obs.append(feature)
        return feature

    def runVIO(self, mode = 0, path_to_output = "./", frames_gt=[], maxtime=-1, bNoiseData = False, iteration = 1):
        """Run VIO for an epoch

        Args:
            mode (int, optional): 0: Linearize observation model without liearization error. Defaults to 0.
        """
        if mode == 0:
            self.runVIOWithoutError(path_to_output, maxtime)
        elif mode == 1:
            return self.runVIOWithoutError_CLS(path_to_output, frames_gt, maxtime, bNoiseData, iteration)
        elif mode == 2:
            return self.runVIOWithoutError_FilterAllState(path_to_output, frames_gt, maxtime, bNoiseData, iteration)
        elif mode == 3:
            return self.runVIOWithoutError_CLS_Sequential(path_to_output, frames_gt, maxtime, bNoiseData, iteration)
        elif mode == 4:
            return self.runVIOWithoutError_FilterAllState_Window(path_to_output, frames_gt, maxtime, bNoiseData)


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

    def runVIOWithoutError_FilterAllState(self, path_to_output, frames_gt, maxtime=-1, bNoiseData = False, iteration=1):
        frames_estimate = self.m_filter.filter_AllState(self.m_frames, self.m_camera, frames_gt, maxtime, iteration)
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
