from operator import inv
from re import L
from time import sleep
from vcommon import *
import numpy as np
from camera import *


class KalmanFilter:
    def __init__(self, PhiPose = np.identity(6), QPose = np.identity(6), QPoint=0.01, PosStd = 0.01, AttStd = 0.01, PointStd = 0.01, PixelStd = 1):
        """Initialize Kalman Filter

        Args:
            PhiPose  (ndarray, optional):   State transition matrix of pose.            Defaults to np.identity((6, 6)).
            QPose    (ndarray, optional):   Noise in state prediction of pose.          Defaults to np.identity((6, 6)).
            QPoint   (float, optional):     Noise in stae prediction of mappoint.       Defaults to 0.01(in meter).
            PosStd   (float, optional):     Standard deviation of position of camera.   Defaults to 0.01(in meter).
            AttStd   (float, optional):     Standard deviation of attitude of camera.   Defaults to 0.01(in degree).
            PointStd (float, optional):     Standard deviation of mappoint of camera.   Defaults to 0.01(in meter).
        """
        self.m_PhiPose = PhiPose
        self.m_QPose = QPose
        self.m_QPoint = QPoint
        self.m_PosStd = PosStd
        self.m_AttStd = AttStd
        self.m_PointStd = PointStd
        self.m_StateCov = np.identity(6)
        self.m_PixelStd = PixelStd

        self.m_StateCov[:3, :3] *= (PosStd )
        self.m_StateCov[3:, 3:] *= (AttStd )

        self.m_MapPoints = {}       # position of mappoint in state vector
        self.m_MapPointPos = 6      # [cameraPos CameraRotation point1 ... pointN]
        self.m_MapPoints_Prev = {}  # previous window position

        self.m_MapPoints_Point = {}
        self.m_MapPoints_Point_linear = {}       # position of mappoint in state vector
        self.m_estimateFrame = []
        self.m_Nmarg = np.zeros((0, 0))
        self.m_Xmarg = np.zeros((0, 0))
        self.m_Dmarg = np.zeros((0, 0))
        self.m_bmarg = np.zeros((0, 0))
        self.m_Jmarg = np.zeros((0, 0))
        self.m_lmarg = np.zeros((0, 0))
        self.m_Npmarg = np.zeros((0, 0))
        self.m_bpmarg = np.zeros((0, 0))
        self.m_Xpmarg = np.zeros((0, 0))
        self.m_Pobsmarg = np.zeros((0, 0))
        self.m_LandmarkLocal = {}
        self.m_FrameFEJ = {}
        self.m_LandmarkFEJ = {}
        

    def reset(self):
        self.m_MapPoints = {}       # position of mappoint in state vector
        self.m_MapPointPos = 0      # [cameraPos CameraRotation point1 ... pointN]
        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []
        self.m_Nmarg = np.zeros((0, 0))
        self.m_bmarg = np.zeros((0, 0))
        self.m_LandmarkLocal = {}
        

    def filter(self, tec, Rec, features, camera):
        """VIO filter for an epoch

        Args:
            tec (ndarray): position of frame, t_{ec}^{e}
            Rec (ndarray): rotation of frame, R_{e}^{c}
            features (list): list of features in image plane
            camera  (class): camera model
        """
        # 1. determine position of mappoint in Jacobian matrix
        obsnum = len(features)
        for i in range(obsnum):
            feature = features[i]
            mappoint = feature.m_mappoint
            mappointID = mappoint.m_id
            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPointPos += 3
        
        # 2. determine size of matrix in KF
        # print(self.m_MapPoints)
        MapPointNumTotal = len(self.m_MapPoints)
        MapPointStateNum = MapPointNumTotal * 3

        Phi = np.identity(MapPointStateNum + 6)     # state transition matrix
        Q = np.identity(MapPointStateNum + 6)      # noise during state transition
        Q[:6, : 6] = self.m_QPose
        Q[6:, 6:] = np.identity(MapPointStateNum) * self.m_QPoint

        R = np.identity(obsnum * 3) * self.m_PixelStd * self.m_PixelStd

        # update new point covariance
        StateCov = np.identity(MapPointStateNum + 6)
        row = self.m_StateCov.shape[0]
        StateCov[:row, :row] = self.m_StateCov
        StateCov[row:, row:] = np.identity(StateCov[row:, row:].shape[0]) * self.m_PointStd * self.m_PointStd

        # 注意： 由于状态转移矩阵为单位阵，因此一步预测的结果与系统状态一致，线性化时不必考虑

        # 3. set jacobian matrix
        J, l = self.setMEQ(tec, Rec, features, camera)

        # 4. do kalman filter
        state_cov_pre = Phi @ StateCov @ Phi.transpose() + Q
        K = state_cov_pre @ J.transpose() @ np.linalg.inv(J @ state_cov_pre @ J.transpose() + R)
        state = K @ l
        self.m_StateCov = (np.identity(K.shape[0]) - K @ J) @ state_cov_pre

        # 5. Error rectify
        tec -= state[:3, :]
        MisAngle = state[3 : 6, :]
        Rec = Rec @ (np.identity(3) - SkewSymmetricMatrix(MisAngle))
        # print(Rec)
        # print(tec.transpose())
        for i in range(obsnum):
            feature = features[i]
            mappoint = feature.m_mappoint
            mappointID = mappoint.m_id
            index = self.m_MapPoints[mappointID]
            mappoint.m_pos -= state[index : index + 3, :]



    def setMEQ(self, tec, Rec, features, camera):
        """Set jacobian matrix

        Args:
            tec (ndarray): frame position in e-frame, t^{e}_{ec}
            Rec (ndarray): rotation from e-frame to c-frame, R_{e}
            features (list): features
            camera  (class): camera model
        
        Returns:
            ndarray:    Jacobain matrix
            ndarray:    residual matrix
        """
        obsnum = len(features)
        stateNum = len(self.m_MapPoints) * 3 + 6
        jaco = np.zeros((obsnum * 3, stateNum))
        l = np.zeros((obsnum * 3, 1))
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b

        for row in range(obsnum):
            feat = features[row]
            mappoint = feat.m_mappoint
            pointID = mappoint.m_id
            pointPos = mappoint.m_pos
            pointPos_c = np.matmul(Rec, (pointPos - tec))
            uv = camera.project(pointPos_c)
            uv_obs = feat.m_pos
            
            l_sub = uv - uv_obs
            l[row * 3: row * 3 + 3, :] = l_sub

            Jphi = Jacobian_phai(Rec, tec, pointPos, pointPos_c, fx, fy, b)
            Jrcam = Jacobian_rcam(Rec, pointPos_c, fx, fy, b)
            JPoint = Jacobian_Point(Rec, pointPos_c, fx, fy, b)

            PointIndex = self.m_MapPoints[pointID]

            jaco[row * 3: row * 3 + 3, 0 : 3] = Jrcam
            jaco[row * 3: row * 3 + 3, 3 : 6] = Jphi
            jaco[row * 3: row * 3 + 3, PointIndex : PointIndex + 3] = JPoint

        return jaco, l

    def filter_AllState(self, frames, camera, frames_gt,frames_linear, maxtime=-1, iteration = 1):
        np.set_printoptions(threshold=np.inf)
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time
        self.m_MapPointPos = 0
        self.m_MapPoints = {}       # position of mappoint in state vector

        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []
        # determine matrix size
        obsnum, statenum = 0, 0
        # count for observations and state
        count = 0
        for frame in frames:
            if frame.m_time > LastTime:
                break
            features = frame.m_features
            obsnum += len(features) * 3
            self.__addFeatures(features)
            self.m_estimateFrame.append(frame)

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        pointStateNum = len(self.m_MapPoints) * 3
        StateFrameNum = len(self.m_estimateFrame) * 6

        MaxIter = iteration
        if iteration == -1:
            MaxIter = 1

        for iter in range(MaxIter):
            # init KF matrix
            Phi = np.identity(statenum)     # state transition matrix
            Q = np.zeros((statenum, statenum))       # noise during state transition
            # Q[:StateFrameNum, : StateFrameNum] = self.m_QPose
            i = 0
            while True:
                Q[i : i + 6, i : i + 6] = self.m_QPose
                i += 6
                if i >= statenum - 6:
                    break

            Q[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * self.m_QPoint
            print(self.m_AttStd)
            self.m_StateCov = np.identity(statenum)
            PoseCov = np.identity(6)
            PoseCov[:3, :3] *= (self.m_PosStd ** 2)
            PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

            i = 0
            while True:
                self.m_StateCov[i: i + 6, i: i + 6] = PoseCov
                i += 6

                if i >= self.m_StateCov.shape[0] - 6:
                    break
            self.m_StateCov[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * (self.m_PointStd ** 2)
            # if iter == 1:
            #     np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/CovFilter.txt", self.m_StateCov)
            #     exit(-1)
            state = np.zeros((statenum, 1))
            for i in range(len(self.m_estimateFrame)):
                gt = frames_gt[i]
                frame = self.m_estimateFrame[i]
                tec, Rec = frame.m_pos, frame.m_rota
                # if iteration == -1 and i != 0:
                #     tec, Rec = self.m_estimateFrame[i - 1].m_pos, self.m_estimateFrame[i - 1].m_rota
                features = frame.m_features

                obsnum = len(features) * 3
                R = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
                J, l = self.setMEQ_AllState(tec, Rec, features, camera, i)
                # l_all = l_all + l
                W = np.linalg.inv(R)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_FILTER_" + str(i) + ".txt", J)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_FILTER_" + str(i) + ".txt", l)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/W_FILTER_" + str(i) + ".txt", W)
                print("Process " + str(i) + "th frame")
                state_cov_pre = Phi @ self.m_StateCov @ Phi.transpose() + Q
                K = state_cov_pre @ J.transpose() @ np.linalg.inv(J @ state_cov_pre @ J.transpose() + R)
                if iteration == -1:
                    state = K @ l
                else:
                    state = state + K @ (l - J @ state)
                # print(state.transpose())
                # original covariance matrix
                tmp = (np.identity(K.shape[0]) - K @ J)
                CovTmp = tmp @ state_cov_pre

                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/COV.txt")
                # test covariance matrix
                # tmp = np.linalg.inv(np.linalg.inv(state_cov_pre) + J.transpose() @ W @ J)
                # CovTmp = tmp
                self.m_StateCov = CovTmp

                if iteration != -1:
                    continue
                # update all frames
                for j in range(len(self.m_estimateFrame)): 
                    self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
                    self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
                # print(state)
                for id_ in self.m_MapPoints.keys():
                    position = self.m_MapPoints[id_]
                    self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

                state = np.zeros((statenum, 1))

            print(state)
            if iteration == -1:
                continue

            # update all frames
            for j in range(len(self.m_estimateFrame)): 
                self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
                self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
            # print(state)
            for id_ in self.m_MapPoints.keys():
                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

        return frames


    def filter_AllState_PPT(self, frames, camera, frames_gt, frames_linear, maxtime=-1, iteration = 1):
        np.set_printoptions(threshold=np.inf)
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time
        self.m_MapPointPos = 0
        self.m_MapPoints = {}       # position of mappoint in state vector

        self.m_MapPoints_Point = {}
        self.m_MapPoints_Point_linear = {}
        self.m_estimateFrame = []
        self.m_estimateFrame_linear = []
        # determine matrix size
        obsnum, statenum = 0, 0
        # count for observations and state
        count = 0
        for i in range(len(frames)):
            frame = frames[i]
            frame_linear = frames_linear[i]
            if frame.m_time > LastTime:
                break
            features = frame.m_features
            features_linear = frame_linear.m_features
            obsnum += len(features) * 3
            self.__addFeatures_PPT(features, features_linear)
            self.m_estimateFrame.append(frame)
            self.m_estimateFrame_linear.append(frame_linear)

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        pointStateNum = len(self.m_MapPoints) * 3
        StateFrameNum = len(self.m_estimateFrame) * 6

        MaxIter = iteration
        if iteration == -1:
            MaxIter = 1

        for iter in range(MaxIter):
            # init KF matrix
            Phi = np.identity(statenum)     # state transition matrix
            Q = np.zeros((statenum, statenum))       # noise during state transition
            # Q[:StateFrameNum, : StateFrameNum] = self.m_QPose
            j = 0
            while True:
                Q[j : j + 6, j : j + 6] = self.m_QPose
                j += 6
                if j >= statenum - 6:
                    break

            Q[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * self.m_QPoint
            print(self.m_AttStd)
            self.m_StateCov = np.identity(statenum)
            PoseCov = np.identity(6)
            PoseCov[:3, :3] *= (self.m_PosStd ** 2)
            PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

            j = 0
            while True:
                self.m_StateCov[j: j + 6, j: j + 6] = PoseCov
                j += 6

                if j >= self.m_StateCov.shape[0] - 6:
                    break
            self.m_StateCov[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * (self.m_PointStd ** 2)
            # if iter == 1:
            #     np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/CovFilter.txt", self.m_StateCov)
            #     exit(-1)
            state = np.zeros((statenum, 1))
            for i in range(len(self.m_estimateFrame)):
                gt = frames_gt[i]
                frame = self.m_estimateFrame_linear[i]
                tec, Rec = frame.m_pos, frame.m_rota
                # if iteration == -1 and i != 0:
                #     tec, Rec = self.m_estimateFrame[i - 1].m_pos, self.m_estimateFrame[i - 1].m_rota
                features = frame.m_features

                obsnum = len(features) * 3
                R = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
                J, l = self.setMEQ_AllState(tec, Rec, features, camera, i)
                # l_all = l_all + l
                W = np.linalg.inv(R)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_FILTER_" + str(i) + ".txt", J)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_FILTER_" + str(i) + ".txt", l)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/W_FILTER_" + str(i) + ".txt", W)
                print("Process " + str(i) + "th frame")
                state_cov_pre = Phi @ self.m_StateCov @ Phi.transpose() + Q
                K = state_cov_pre @ J.transpose() @ np.linalg.inv(J @ state_cov_pre @ J.transpose() + R)
                if iteration == -1:
                    state = K @ l
                else:
                    state = K @ l # state + K @ (l - J @ state)
                # print(state.transpose())
                # original covariance matrix
                tmp = (np.identity(K.shape[0]) - K @ J)
                CovTmp = tmp @ state_cov_pre

                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/COV.txt")
                # test covariance matrix
                # tmp = np.linalg.inv(np.linalg.inv(state_cov_pre) + J.transpose() @ W @ J)
                # CovTmp = tmp
                self.m_StateCov = CovTmp

                # if iteration != -1:
                #     continue
                # update all frames
                for j in range(len(self.m_estimateFrame)): 
                    self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
                    self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
                # print(state)
                for id_ in self.m_MapPoints.keys():
                    position = self.m_MapPoints[id_]
                    self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

                state = np.zeros((statenum, 1))

            # print(state)
            if iteration == -1:
                continue
            
            for i in range(len(self.m_estimateFrame)):
                frame = self.m_estimateFrame[i]
                frame_linear = self.m_estimateFrame_linear[i]

                frame_linear.m_pos = frame.m_pos.copy()
                frame_linear.m_rota = frame.m_rota.copy()

            for id in self.m_MapPoints_Point.keys():
                point = self.m_MapPoints_Point[id]
                point_linear = self.m_MapPoints_Point_linear[id]

                point_linear.m_pos = point.m_pos.copy()
            # # update all frames
            # for j in range(len(self.m_estimateFrame)): 
            #     self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
            #     self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
            # # print(state)
            # for id_ in self.m_MapPoints.keys():
            #     position = self.m_MapPoints[id_]
            #     self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

        return frames


    def filter_AllState_Window(self, frames, camera, frames_gt, maxtime=-1):
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time
        self.m_MapPointPos = 0
        self.m_MapPoints = {}       # position of mappoint in state vector

        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []
        # determine matrix size
        obsnum, statenum = 0, 0
        # count for observations and state
        count = 0

        for frame in frames:
            if frame.m_time > LastTime:
                break
            features = frame.m_features
            obsnum += len(features) * 3
            self.__addFeatures(features)
            self.m_estimateFrame.append(frame)

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        pointStateNum = len(self.m_MapPoints) * 3
        StateFrameNum = len(self.m_estimateFrame) * 6

        # init KF matrix
        Phi = np.identity(statenum)     # state transition matrix
        Q = np.zeros((statenum, statenum))       # noise during state transition
        # Q[:StateFrameNum, : StateFrameNum] = self.m_QPose
        i = 0
        while True:
            Q[i : i + 6, i : i + 6] = self.m_QPose
            i += 6
            if i >= statenum - 6:
                break

        Q[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * self.m_QPoint
        print(self.m_AttStd)
        self.m_StateCov = np.identity(statenum)
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        B, L = np.zeros((obsnum, statenum)), np.zeros((obsnum, 1))

        i = 0
        while True:
            self.m_StateCov[i: i + 6, i: i + 6] = PoseCov
            i += 6

            if i >= self.m_StateCov.shape[0] - 6:
                break
        self.m_StateCov[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * (self.m_PointStd ** 2)
        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/CovFilter.txt", self.m_StateCov)
        obsnum_all = 0
        for i in range(len(self.m_estimateFrame)):
            gt = frames_gt[i]
            frame = self.m_estimateFrame[i]
            tec, Rec = gt.m_pos, gt.m_rota
            features = gt.m_features
            obsnum_sub = len(features) * 3
            # R = np.identity(obsnum_sub) * self.m_PixelStd * self.m_PixelStd
            J, l = self.setMEQ_AllState(tec, Rec, features, camera, i)
            # W = np.linalg.inv(R)

            B[obsnum_all: obsnum_all + obsnum_sub, :] = J
            L[obsnum_all: obsnum_all + obsnum_sub, :] = l
            obsnum_all += obsnum_sub
            print("Process " + str(i) + "th frame")
        print("Start to filter")
        state_cov_pre = Phi @ self.m_StateCov @ Phi.transpose() + Q
        R = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
        K = state_cov_pre @ B.transpose() @ np.linalg.inv(B @ state_cov_pre @ B.transpose() + R)
        state = K @ L
        # original covariance matrix
        tmp = (np.identity(K.shape[0]) - K @ B)
        CovTmp = tmp @ state_cov_pre
        self.m_StateCov = CovTmp
        print("filter done")
        # update current frame
        # FramedX = state[i * 6: i * 6 + 6, :]
        # self.m_estimateFrame[i].m_pos = self.m_estimateFrame[i].m_pos - FramedX[0: 3]
        # self.m_estimateFrame[i].m_rota = self.m_estimateFrame[i].m_rota @ (np.identity(3) - SkewSymmetricMatrix(FramedX[3: 6]))

        # for feat in features:
        #     MapPointID = feat.m_mappoint.m_id
        #     MapPointPos = self.m_MapPoints[MapPointID]
        #     self.m_MapPoints_Point[MapPointID].m_pos = self.m_MapPoints_Point[MapPointID].m_pos - state[StateFrameNum + MapPointPos: StateFrameNum + MapPointPos + 3, :]

        # update all frames
        for j in range(len(self.m_estimateFrame)): 
            self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
            self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

        return frames

    def setMEQ_AllState(self, tec, Rec, features, camera, frame_i):

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        obsnum = len(features)
        FrameStateNum = len(self.m_estimateFrame) * 6

        J, l = np.zeros((obsnum * 3, statenum)), np.zeros((obsnum * 3, 1))
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b
        for row in range(obsnum):
            feat = features[row]
            mappoint = feat.m_mappoint
            pointID = mappoint.m_id
            pointPos = mappoint.m_pos
            pointPos_c = np.matmul(Rec, (pointPos - tec))
            uv = camera.project(pointPos_c)
            uv_obs = feat.m_pos
            PointIndex = self.m_MapPoints[pointID]
            
            l_sub = uv - uv_obs
            l[row * 3: row * 3 + 3, :] = l_sub

            Jphi = Jacobian_phai(Rec, tec, pointPos, pointPos_c, fx, fy, b)
            Jrcam = Jacobian_rcam(Rec, pointPos_c, fx, fy, b)
            JPoint = Jacobian_Point(Rec, pointPos_c, fx, fy, b)

            J[row * 3: row * 3 + 3, frame_i * 6 : frame_i * 6 + 3] = Jrcam
            J[row * 3: row * 3 + 3, frame_i * 6 + 3 : frame_i * 6 + 6] = Jphi
            J[row * 3: row * 3 + 3, FrameStateNum + PointIndex : FrameStateNum + PointIndex + 3] = JPoint

        return J, l

    def solveSW(self, frames, frames_gt, camera, windowsize=20, maxtime=-1, iteration=1):
        """_summary_

        Args:
            frames (_type_): _description_
            frames_gt (_type_): _description_
            camera (_type_): _description_
            windowsize (int, optional): _description_. Defaults to 20.
            maxtime (int, optional): _description_. Defaults to -1.
            iteration (int, optional): _description_. Defaults to 1.
        """
        StateFrameSize = windowsize * 6
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time
        
        # initialize sliding window
        LocalFrames, LocalFrames_gt = {}, {}
        self.m_StateCov = np.zeros((StateFrameSize, StateFrameSize))
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        i = 0
        while True:
            self.m_StateCov[i: i + 6, i: i + 6] = PoseCov
            i += 6

            if i >= self.m_StateCov.shape[0]:
                break

        Local = 0
        StateFrame = np.zeros((windowsize * 6, 1))
        for i in range(len(frames)):
            if frames[i].m_time > LastTime:
                break
            LocalFrames[Local] = frames[i]
            LocalFrames_gt[Local] = frames_gt[i]

            Local += 1
            if Local < windowsize:
                continue
            tmp = (windowsize - 1) * 6
            self.m_StateCov[tmp: , tmp:, ] = PoseCov

            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId, frame in LocalFrames.items():
                nobs += len(frame.m_features) * 3
                self.__addFeatures(frame.m_features)
            StateLandmark = len(self.m_MapPoints) * 3
            print("process " + str(i) + "th frame. Landmark: " + str(len(self.m_MapPoints)) + ", observation num: " + str(nobs / 3) + ", Local frame size: " + str(len(LocalFrames)))

            #TODO: 1. solve CLS problem by marginalizing landmark
            AllStateNum = windowsize * 6 + StateLandmark
            TotalObsNum = 0
            B, L = np.zeros((nobs, AllStateNum)), np.zeros((nobs, 1))
            for LocalID, frame in LocalFrames.items():
                tec, Rec = frame.m_pos, frame.m_rota
                features = frame.m_features
                obsnum = len(features) * 3
                J, l = self.setMEQ_SW(tec, Rec, features, camera, windowsize, LocalID)

                B[TotalObsNum : TotalObsNum + obsnum, :] = J
                L[TotalObsNum : TotalObsNum + obsnum, :] = l
                TotalObsNum += obsnum
            B_all, L_all = np.zeros((nobs + windowsize * 6, AllStateNum)), np.zeros((nobs + windowsize * 6, 1))
            P_all = np.zeros((windowsize * 6 + nobs, windowsize * 6 + nobs))

            # prior part
            B_all[: windowsize * 6, :windowsize * 6] = np.identity(windowsize * 6)
            P_all[: windowsize * 6, :windowsize * 6] = self.m_StateCov
            L_all[: windowsize * 6, :] = StateFrame #WARNING: bugs may remain if not rectifying errors

            # observation part
            B_all[windowsize * 6:, ] = B
            P_all[windowsize * 6:, windowsize * 6: ] = np.identity(nobs) * self.m_PixelStd * self.m_PixelStd
            L_all[windowsize * 6:, ] = L
            P_all = np.linalg.inv(P_all)

            N = B_all.transpose() @ P_all @ B_all
            b = B_all.transpose() @ P_all @ L_all

            N11 = N[: windowsize * 6, : windowsize * 6]
            N12 = N[: windowsize * 6, windowsize * 6: ]
            N22 = N[windowsize * 6: , windowsize * 6: ]
            b1  = b[: windowsize * 6, : ]
            b2  = b[windowsize * 6: , : ]
            N22_inv = np.linalg.inv(N22)


            self.m_StateCov = np.linalg.inv(N11 - N12 @ N22_inv @ N12.transpose())
            state = np.linalg.inv(N11 - N12 @ N22_inv @ N12.transpose()) @ (b1 - N12 @ N22_inv @ b2)
            StateFrame = state[:, :]
            # print(state)


            # 2. update states. evaluate jacobian at groundtruth, do not update.
            for j in range(Local):  
                LocalFrames[j].m_pos = LocalFrames[j].m_pos - state[j * 6: j * 6 + 3, :]
                LocalFrames[j].m_rota = LocalFrames[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

            # for id_ in self.m_MapPoints.keys():
            #     position = self.m_MapPoints[id_]
            #     self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

            # 3. remove old frame and its covariance
            for _id in range(Local - 1):
                LocalFrames_gt[_id] = LocalFrames_gt[_id + 1]
                LocalFrames[_id] = LocalFrames[_id + 1]
            tmp = (windowsize - 1) * 6
            self.m_StateCov[: tmp, : tmp] = self.m_StateCov[6: , 6: ]
            self.m_StateCov[tmp:, :] = 0
            self.m_StateCov[:, tmp: ] = 0
            Local -= 1
            StateFrame[: tmp, :] = StateFrame[6:, :]
            StateFrame[tmp:, :] = 0
            # print(StateFrame)
        return frames

    def solveSW_Marg(self, frames, frames_gt, camera, windowsize=20, maxtime=-1, iteration=1):
        """_summary_

        Args:
            frames (_type_): _description_
            frames_gt (_type_): _description_
            camera (_type_): _description_
            windowsize (int, optional): _description_. Defaults to 20.
            maxtime (int, optional): _description_. Defaults to -1.
            iteration (int, optional): _description_. Defaults to 1.
        """
        StateFrameSize = windowsize * 6
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time
        
        # initialize sliding window
        LocalFrames, LocalFrames_gt = {}, {}
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        Local = 0
        StateFrame = np.zeros((windowsize * 6, 1))
        State = np.zeros((windowsize * 6, 1))
        for i in range(len(frames)):
            if frames[i].m_time > LastTime:
                break
            LocalFrames[Local] = frames[i]
            LocalFrames_gt[Local] = frames_gt[i]

            Local += 1
            if Local < windowsize:
                continue

            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId, frame in LocalFrames.items():
                nobs += len(frame.m_features) * 3
                self.__addFeatures(frame.m_features)
            StateLandmark = len(self.m_MapPoints) * 3
            print("process " + str(i) + "th frame. Landmark: " + str(len(self.m_MapPoints)) + ", observation num: " + str(nobs / 3) + ", Local frame size: " + str(len(LocalFrames)))
            
            AllStateNum = windowsize * 6 + StateLandmark
            TotalObsNum = 0
            self.m_StateCov = np.zeros((StateFrameSize + StateLandmark, StateFrameSize + StateLandmark))
            j = 0
            while True:
                self.m_StateCov[j: j + 6, j: j + 6] = PoseCov
                j += 6

                if j >= StateFrameSize:
                    break
            self.m_StateCov[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
            tmp = (windowsize - 1) * 6
            self.m_StateCov[tmp: StateFrameSize, tmp: StateFrameSize] = PoseCov

            # 1. solve CLS problem by marginalizing landmark
            B, L = np.zeros((nobs, AllStateNum)), np.zeros((nobs, 1))
            # N_obs, b_obs = np.zeros((AllStateNum, AllStateNum)), np.zeros((AllStateNum, 1))
            for LocalID, frame in LocalFrames.items():
                tec, Rec = frame.m_pos, frame.m_rota
                features = frame.m_features
                obsnum = len(features) * 3

                J, l = self.setMEQ_SW(tec, Rec, features, camera, windowsize, LocalID)

                B[TotalObsNum : TotalObsNum + obsnum, :] = J
                L[TotalObsNum : TotalObsNum + obsnum, :] = l
                TotalObsNum += obsnum


            NPrior_inv, XPrior = self.CovConstraint(windowsize, AllStateNum)
            dx = self.compensateFEJ(windowsize)
            P_obs = np.identity(nobs) * (self.m_PixelStd * self.m_PixelStd)
            K = NPrior_inv @ B.transpose() @ np.linalg.inv(P_obs + B @ NPrior_inv @ B.transpose())
            # XPrior = NPrior_inv @ bPrior
            # print(np,all(np.abs(XPrior1 - XPrior) < 1E-5))
            state = dx + XPrior + K @ (L - B @ (XPrior + dx))
            StateFrame = state[: windowsize * 6]

            # update covariance
            self.UpdateCov(LocalFrames, NPrior_inv, camera, windowsize, XPrior)

            # 2. update states. evaluate jacobian at groundtruth, do not update.
            for j in range(Local):  
                LocalFrames[j].m_pos = LocalFrames[j].m_pos - state[j * 6: j * 6 + 3, :]
                LocalFrames[j].m_rota = LocalFrames[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
            StateFrameNum = windowsize * 6

            for id_ in self.m_MapPoints.keys():
                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3, :]

            # 3. remove old frame and its covariance
            for _id in range(Local - 1):
                LocalFrames_gt[_id] = LocalFrames_gt[_id + 1]
                LocalFrames[_id] = LocalFrames[_id + 1]
            tmp = (windowsize - 1) * 6
            self.m_StateCov[: tmp, : tmp] = self.m_StateCov[6: StateFrameSize, 6: StateFrameSize]
            self.m_StateCov[tmp:, :] = 0
            self.m_StateCov[:, tmp: ] = 0
            Local -= 1
            StateFrame[: tmp, :] = StateFrame[6: StateFrameSize, :]
            StateFrame[tmp: StateFrameSize, :] = 0
            self.m_MapPoints_Prev = self.m_MapPoints
        return frames
    
    def CovConstraint(self, windowsize, StateNum):
        """Prepare prior information produced by marginalization
        """
        StateFrameSize = windowsize * 6
        StateLandmark = len(self.m_MapPoints) * 3

        NPrior, NPrior_inv, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
        state_prior = np.zeros((StateNum, 1))
        # set diagnoal to micro-value
        mapping = {}
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        j = 0
        while True:
            NPrior_inv[j: j + 6, j: j + 6] = PoseCov
            j += 6

            if j >= StateFrameSize:
                break
        NPrior_inv[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
        NPrior = np.linalg.inv(NPrior_inv)

        if len(self.m_LandmarkLocal) == 0:
            B, L = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            P = np.zeros((StateNum, StateNum))
            B = np.identity(StateNum)
            P = NPrior

            L[: windowsize * 6, :] = 0 
            # print(L)
            bPrior = B.transpose() @ P @ L
        else:
            Dmarg, dx = self.submarg()
            FrameStateNum = windowsize * 6
            mapping = {}
            # NPrior, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            for mappointID, GlobalPos in self.m_MapPoints.items():
                if mappointID in self.m_LandmarkLocal.keys():
                    LocalPos = self.m_LandmarkLocal[mappointID] * 3
                    mapping[GlobalPos + FrameStateNum] = LocalPos
            for gpos, lpos in mapping.items():
                for gpos1, lpos1 in mapping.items():
                    # NPrior[gpos: gpos + 3, gpos1: gpos1 + 3] = Nmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                    NPrior_inv[gpos: gpos + 3, gpos1: gpos1 + 3] = Dmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                # bPrior[gpos: gpos + 3, : ] = bmarg[lpos: lpos + 3, :]
                state_prior[gpos: gpos + 3, : ] = dx[lpos: lpos + 3, :]

        return NPrior_inv, state_prior #, NPrior_inv, X_return, dX


    def CovConstraint_Kitti(self, windowsize, StateNum):
        """Prepare prior information produced by marginalization
        """
        StateFrameSize = windowsize * 6
        StateLandmark = len(self.m_MapPoints) * 3

        NPrior, NPrior_inv, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
        # set diagnoal to micro-value
        mapping = {}
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        j = 0
        while True:
            NPrior_inv[j: j + 6, j: j + 6] = PoseCov
            j += 6

            if j >= StateFrameSize:
                break
        NPrior_inv[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
        NPrior = np.linalg.inv(NPrior_inv)

        if len(self.m_LandmarkLocal) == 0:
            B, L = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            P = np.zeros((StateNum, StateNum))
            B = np.identity(StateNum)
            P = NPrior

            L[: windowsize * 6, :] = 0 
            # print(L)
            NPrior = B.transpose() @ P @ B
            bPrior = B.transpose() @ P @ L
        else:
            bmarg, Nmarg = self.submarg_Kitti()
            FrameStateNum = windowsize * 6
            mapping = {}
            # NPrior, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            for mappointID, GlobalPos in self.m_MapPoints.items():
                if mappointID in self.m_LandmarkLocal.keys():
                    LocalPos = self.m_LandmarkLocal[mappointID] * 3
                    mapping[GlobalPos + FrameStateNum] = LocalPos
            for gpos, lpos in mapping.items():
                for gpos1, lpos1 in mapping.items():
                    NPrior[gpos: gpos + 3, gpos1: gpos1 + 3] = Nmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                    # NPrior_inv[gpos: gpos + 3, gpos1: gpos1 + 3] = Dmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                bPrior[gpos: gpos + 3, : ] = bmarg[lpos: lpos + 3, :]

        return bPrior, np.linalg.inv(NPrior) #, NPrior_inv, X_return, dX

    def marginalization(self, WindowSize, LocalFrame, camera, NPrior, bPrior):

        FirstLocalID = list(LocalFrame.keys())[0]
        frame = LocalFrame[FirstLocalID]
        tec, Rec = frame.m_pos, frame.m_rota
        features = frame.m_features

        # step 1: select to-be-removed states and measurements
        J, l = self.setMEQ_SW(tec, Rec, features, camera, WindowSize, FirstLocalID)
        P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        N = J.transpose() @ P_obs @ J
        HaveValue = np.diagonal(N) != 0
        N += NPrior

        N = N[[HaveValue[i] for i in range(len(HaveValue))], :]
        N_sub = N[:, [HaveValue[i] for i in range(len(HaveValue))]]

        b_all = J.transpose() @ P_obs @ l + bPrior
        b_sub = b_all[[HaveValue[i] for i in range(len(HaveValue))], :]
        
        # step 2: marginalization
        N_marg = N_sub # N22 - N12_T @ N11_inv @ N12
        b_marg = b_sub # b2 - N12_T @ N11_inv @ b1

        # step 3: specify map point ID -- position in N_marg
        items_to_remove = []
        for mappointID, value in self.m_LandmarkFEJ.items():
            if mappointID not in self.m_MapPoints.keys():
                items_to_remove.append(mappointID)
        
        for item in items_to_remove:
            del self.m_LandmarkFEJ[item]

        PosLandmarkStart = WindowSize * 6
        ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkLocal = {}
        for i in range(len(ConnectedNodes)):
            for mappointID, value in self.m_MapPoints.items():
                if ConnectedNodes[i] * 3 == value:
                    LandmarkLocal[mappointID] = i
                    # fix linearization point
                    if mappointID not in self.m_LandmarkFEJ.keys():
                        self.m_LandmarkFEJ[mappointID] = copy.deepcopy(self.m_MapPoints_Point[mappointID].m_pos)
                    break

        self.m_Nmarg = N_marg
        self.m_bmarg = b_marg
        self.m_LandmarkLocal = LandmarkLocal

    def compensateFEJ(self, windowsize=20):
        Ncom, bcom = 0, 0
        if len(self.m_LandmarkLocal) == 0:
            return 0

        FrameStateNum, StateLandmark = windowsize * 6, len(self.m_MapPoints) * 3
        mapping = {}

        for mappointID, GlobalPos in self.m_MapPoints.items():
            if mappointID in self.m_LandmarkLocal.keys():
                LocalPos = self.m_LandmarkLocal[mappointID] * 3
                mapping[GlobalPos + FrameStateNum] = LocalPos
        if self.m_Jmarg.shape[0] != 0:
            # differences of constrained landmarks.
            # note that frames are not constrained in this case
            # frame of Xdiff remains 0
            Xdiff = np.zeros((FrameStateNum + StateLandmark, 1))
            for mappointID, point in self.m_MapPoints_Point.items():
                pos = self.m_MapPoints[mappointID] + FrameStateNum
                point_FEJ = self.getLandmarkFEJ(point)
                Xdiff[pos: pos + 3, :] = point.m_pos - point_FEJ
                # print(np.linalg.norm(Xdiff[pos: pos + 3, :]))
            
        return Xdiff

    def getLandmarkFEJ(self, landmark):
        id_to_find = landmark.m_id
        if id_to_find in self.m_LandmarkFEJ.keys():
            return self.m_LandmarkFEJ[id_to_find]
        else:
            # landmark_to_return = copy.deepcopy(landmark)
            # self.m_LandmarkFEJ[id_to_find] = landmark_to_return
            return landmark.m_pos.copy()

    def premarginalization_CLSTEST(self, windowsize, StateNum, StateFrame):
        """Prepare prior information produced by marginalization
        """
        StateFrameSize = windowsize * 6
        StateLandmark = len(self.m_MapPoints) * 3

        NPrior, NPrior_inv, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
        # set diagnoal to micro-value
        mapping = {}
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        j = 0
        while True:
            NPrior_inv[j: j + 6, j: j + 6] = PoseCov
            j += 6

            if j >= StateFrameSize:
                break
        NPrior_inv[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
        NPrior = np.linalg.inv(NPrior_inv)

        if len(self.m_LandmarkLocal) == 0:
            B, L = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            P = np.zeros((StateNum, StateNum))
            B = np.identity(StateNum)
            P = NPrior

            L[: windowsize * 6, :] = StateFrame 
            # print(L)
            NPrior = B.transpose() @ P @ B
            bPrior = B.transpose() @ P @ L
            # NPrior_inv = self.m_StateCov.copy()
        else:
            Nmarg, bmarg = self.submarg_CSLTEST()
            Dmarg = np.linalg.inv(Nmarg)
            FrameStateNum = windowsize * 6
            mapping = {}
            # NPrior, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            for mappointID, GlobalPos in self.m_MapPoints.items():
                if mappointID in self.m_LandmarkLocal.keys():
                    LocalPos = self.m_LandmarkLocal[mappointID] * 3
                    mapping[GlobalPos + FrameStateNum] = LocalPos
            for mappointID in self.m_LandmarkLocal.keys():
                if mappointID not in self.m_MapPoints.keys():
                    print("test")
            for gpos, lpos in mapping.items():
                for gpos1, lpos1 in mapping.items():
                    NPrior[gpos: gpos + 3, gpos1: gpos1 + 3] = Nmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                    NPrior_inv[gpos: gpos + 3, gpos1: gpos1 + 3] = Dmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                bPrior[gpos: gpos + 3, : ] = bmarg[lpos: lpos + 3, :]
            
        return NPrior, bPrior, NPrior_inv#, NPrior_inv, X_return, dX

    def premarginalization(self, windowsize, StateNum, StateFrame):
        """Prepare prior information produced by marginalization
        """
        StateFrameSize = windowsize * 6
        StateLandmark = len(self.m_MapPoints) * 3

        NPrior, NPrior_inv, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
        # set diagnoal to micro-value
        mapping = {}
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        j = 0
        while True:
            NPrior_inv[j: j + 6, j: j + 6] = PoseCov
            j += 6

            if j >= StateFrameSize:
                break
        NPrior_inv[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
        NPrior = np.linalg.inv(NPrior_inv)

        if len(self.m_LandmarkLocal) == 0:
            B, L = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            P = np.zeros((StateNum, StateNum))
            B = np.identity(StateNum)
            P = np.linalg.inv(self.m_StateCov)

            L[: windowsize * 6, :] = StateFrame 
            # print(L)
            NPrior = B.transpose() @ P @ B
            bPrior = B.transpose() @ P @ L
            NPrior_inv = self.m_StateCov.copy()
        else:
            Nmarg, bmarg, Dmarg = self.submarg()
            FrameStateNum = windowsize * 6
            mapping = {}
            # NPrior, bPrior = np.zeros((StateNum, StateNum)), np.zeros((StateNum, 1))
            for mappointID, GlobalPos in self.m_MapPoints.items():
                if mappointID in self.m_LandmarkLocal.keys():
                    LocalPos = self.m_LandmarkLocal[mappointID] * 3
                    mapping[GlobalPos + FrameStateNum] = LocalPos
            for gpos, lpos in mapping.items():
                for gpos1, lpos1 in mapping.items():
                    NPrior[gpos: gpos + 3, gpos1: gpos1 + 3] = Nmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                    NPrior_inv[gpos: gpos + 3, gpos1: gpos1 + 3] = Dmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                bPrior[gpos: gpos + 3, : ] = bmarg[lpos: lpos + 3, :]
            
        return NPrior, bPrior, NPrior_inv 
        

    def submarg(self):
        """do marginalization here to avoid effects caused by iteration

        """

        J, l, Np, bp = self.m_Jmarg, self.m_lmarg, self.m_Npmarg, self.m_bpmarg

        # check observability of landmarks
        LandmarkRemove,  LandmarkRemoveID= [], []
        for mappointid, localpos in self.m_LandmarkLocal.items():
            if mappointid not in self.m_MapPoints.keys():
                LandmarkRemoveID.append(mappointid)
                for i in range(3):
                    LandmarkRemove.append(localpos * 3 + i + 6)

        J = J[:, [i not in LandmarkRemove for i in range(J.shape[1])]]
        Np = Np[[i not in LandmarkRemove for i in range(Np.shape[0])], :]
        Np = Np[:, [i not in LandmarkRemove for i in range(Np.shape[1])]]
        bp = bp[[i not in LandmarkRemove for i in range(bp.shape[0])], :]

        for id in LandmarkRemoveID:
            if id in self.m_LandmarkLocal.keys():
                del self.m_LandmarkLocal[id]
        pos = 0
        for id in self.m_LandmarkLocal.keys():
            self.m_LandmarkLocal[id] = pos
            pos += 1
        
        P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        NPrior_inv = np.linalg.inv(Np)

        K = NPrior_inv @ J.transpose() @ np.linalg.inv(np.linalg.inv(P_obs) + J @ NPrior_inv @ J.transpose())
        Dmarg = (np.identity(K.shape[0]) - K @ J) @ NPrior_inv
        Xp = NPrior_inv @ bp
        dx = Xp + K @ (l - J @ Xp)
        self.m_Jmarg, self.m_lmarg, self.m_bpmarg, self.m_Xpmarg = J, l, Np, bp

        return Dmarg[6:, 6: ], dx[6:, :]


    def submarg_Kitti(self):
        """do marginalization here to avoid effects caused by iteration

        """

        J, l, Np, bp = self.m_Jmarg, self.m_lmarg, self.m_Npmarg, self.m_bpmarg
        P_obs = self.m_Pobsmarg

        # check observability of landmarks
        LandmarkRemove,  LandmarkRemoveID= [], []
        for mappointid, localpos in self.m_LandmarkLocal.items():
            if mappointid not in self.m_MapPoints.keys():
                LandmarkRemoveID.append(mappointid)
                for i in range(3):
                    LandmarkRemove.append(localpos * 3 + i + 6)

        J = J[:, [i not in LandmarkRemove for i in range(J.shape[1])]]
        # P_obs = P_obs[[i not in LandmarkRemove for i in range(P_obs.shape[0])], :]
        # P_obs = P_obs[:, [i not in LandmarkRemove for i in range(P_obs.shape[1])]]
        Np = Np[[i not in LandmarkRemove for i in range(Np.shape[0])], :]
        Np = Np[:, [i not in LandmarkRemove for i in range(Np.shape[1])]]
        bp = bp[[i not in LandmarkRemove for i in range(bp.shape[0])], :]

        # for index in LandmarkRemove:
        #     id = index[0]
        #     J = np.delete(J, index[1], 1)
        #     Np = np.delete(Np, index[1], 0)
        #     Np = np.delete(Np, index[1], 1)
        #     bp = np.delete(bp, index[1], 0)
        # self.m_LandmarkLocal1 = copy.deepcopy(self.m_LandmarkLocal)
        for id in LandmarkRemoveID:
            if id in self.m_LandmarkLocal.keys():
                del self.m_LandmarkLocal[id]
        pos = 0
        for id in self.m_LandmarkLocal.keys():
            self.m_LandmarkLocal[id] = pos
            pos += 1
        
        # P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        NPrior_inv = np.linalg.inv(Np)
        # NPrior_inv = np.linalg.inv(Np + J.transpose() @ P_obs @ J)

        K = NPrior_inv @ J.transpose() @ np.linalg.inv(np.linalg.inv(P_obs) + J @ NPrior_inv @ J.transpose())
        Dmarg = (np.identity(K.shape[0]) - K @ J) @ NPrior_inv

        # b
        N_sub, b_sub = J.transpose() @ P_obs @ J + Np, J.transpose() @ P_obs @ l + bp
        N11, N12 = N_sub[: 6, : 6], N_sub[: 6, 6: ]
        b1, b2 = b_sub[: 6, :], b_sub[6:, :]

        N12_T = N12.transpose()
        N11_inv = np.linalg.inv(N11)
        bmarg = b2 - N12_T @ N11_inv @ b1

        self.m_Jmarg, self.m_lmarg, self.m_Npmarg, self.m_bpmarg = J, l, Np, bp
        self.m_Pobsmarg = P_obs
        return bmarg, np.linalg.inv(Dmarg[6:, 6: ])

    def submarg_CSLTEST(self):
        N_sub, b_sub = self.m_Nmarg, self.m_bmarg
        # check observability of landmarks
        # ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkRemove,  LandmarkRemoveID= [], []
       
        for mappointid, localpos in self.m_LandmarkLocal.items():
            if mappointid not in self.m_MapPoints.keys():
                LandmarkRemoveID.append(mappointid)
                for i in range(3):
                    LandmarkRemove.append(localpos * 3 + i + 6)
        
        N_sub = N_sub[[i not in LandmarkRemove for i in range(N_sub.shape[0])], :]
        N_sub = N_sub[:, [i not in LandmarkRemove for i in range(N_sub.shape[1])]]
        b_sub = b_sub[[i not in LandmarkRemove for i in range(b_sub.shape[0])], :]

        # for index in LandmarkRemove:
        #     id = index[0]
        #     N_sub = np.delete(N_sub, index[1], 0)
        #     N_sub = np.delete(N_sub, index[1], 1)
        #     b_sub = np.delete(b_sub, index[1], 0)

        for id in LandmarkRemoveID:
            if id in self.m_LandmarkLocal.keys():
                del self.m_LandmarkLocal[id]
        pos = 0
        for id in self.m_LandmarkLocal.keys():
            self.m_LandmarkLocal[id] = pos
            pos += 1
        
        N11, N22, N12 = N_sub[: 6, : 6], N_sub[6:, 6: ], N_sub[: 6, 6: ]
        b1, b2 = b_sub[: 6, :], b_sub[6:, :]

        N12_T = N12.transpose()
        N11_inv = np.linalg.inv(N11)
        Nmarg = N22 - N12_T @ N11_inv @ N12
        bmarg = b2 - N12_T @ N11_inv @ b1

        self.m_Nmarg, self.m_bmarg = N_sub, b_sub
        return Nmarg, bmarg


    def __addFeatures(self, features):
        for i in range(len(features)):
            feat = features[i]

            mappoint = feat.m_mappoint
            mappointID = mappoint.m_id

            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
            self.m_MapPointPos += 3


    def __addFeatures_PPT(self, features, features_linear):
        for i in range(len(features)):
            feat = features[i]
            feat_linear = features_linear[i]
            mappoint = feat.m_mappoint
            mappointID = mappoint.m_id

            mappoint_linear = feat_linear.m_mappoint
            mappointID_linear = mappoint_linear.m_id

            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
            self.m_MapPoints_Point_linear[mappointID_linear] = mappoint_linear
            self.m_MapPointPos += 3

    def setMEQ_SW(self, tec, Rec, features, camera, windowsize, LocalID):

        statenum = windowsize * 6 + len(self.m_MapPoints) * 3
        obsnum = len(features)
        FrameStateNum = windowsize * 6

        J, l = np.zeros((obsnum * 3, statenum)), np.zeros((obsnum * 3, 1))
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b

        for row in range(obsnum):
            feat = features[row]
            mappoint = feat.m_mappoint
            pointPos = self.getLandmarkFEJ(mappoint)
            pointID = mappoint.m_id
            # pointPos = mappoint.m_pos
            pointPos_c = np.matmul(Rec, (pointPos - tec))
            uv = camera.project(pointPos_c)
            uv_obs = feat.m_pos
            PointIndex = self.m_MapPoints[pointID]
            
            l_sub = uv - uv_obs
            l[row * 3: row * 3 + 3, :] = l_sub

            Jphi = Jacobian_phai(Rec, tec, pointPos, pointPos_c, fx, fy, b)
            Jrcam = Jacobian_rcam(Rec, pointPos_c, fx, fy, b)
            JPoint = Jacobian_Point(Rec, pointPos_c, fx, fy, b)

            J[row * 3: row * 3 + 3, LocalID * 6 : LocalID * 6 + 3] = Jrcam
            J[row * 3: row * 3 + 3, LocalID * 6 + 3 : LocalID * 6 + 6] = Jphi
            J[row * 3: row * 3 + 3, FrameStateNum + PointIndex : FrameStateNum + PointIndex + 3] = JPoint

        return J, l

    def solveKitti(self, map, camera, windowsize=20):
        frames = map.m_frames
        mappoints = map.m_points

        if len(frames) < 2:
            return 
        windowsize_tmp = len(frames)

        StateFrameSize = len(frames) * 6
        PoseCov = np.identity(6)
        PoseCov[:3, :3] *= (self.m_PosStd ** 2)
        PoseCov[3:, 3:] *= (self.m_AttStd ** 2)

        SaveFrames = None

        prevstate = None
        iter = 0
        while iter < 10:
            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId in range(len(frames)):
                frame = frames[LocalId]
                # nobs += len(frame.m_features) * 3
                nobs += self.__addFeaturesKitti(frame.m_features) * 3
            StateLandmark = len(self.m_MapPoints) * 3
            self.savestates(frames, self.m_MapPoints_Point)

            # 1. solve CLS problem by marginalizing landmark
            AllStateNum = windowsize_tmp * 6 + StateLandmark
            TotalObsNum = 0
            # N, b = np.zeros((AllStateNum, AllStateNum)), np.zeros((AllStateNum, 1))
            R = np.zeros((nobs, nobs))
            B, L = np.zeros((nobs, AllStateNum)), np.zeros((nobs, 1))
            StateFrame = np.zeros((windowsize_tmp * 6, 1))


            for LocalID in range(len(frames)):
                frame = frames[LocalID]
                tec, Rec = frame.m_pos, frame.m_rota
                features = frame.m_features
                
                J, P_obs, l = self.setMEQ_Kitti(tec, Rec, features, camera, windowsize_tmp, LocalID)
                obsnum = l.shape[0]

                B[TotalObsNum : TotalObsNum + obsnum, :] = J
                L[TotalObsNum : TotalObsNum + obsnum, :] = l
                R[TotalObsNum: TotalObsNum + obsnum, TotalObsNum: TotalObsNum + obsnum] = P_obs

                if obsnum <= 9:
                    return -1
                # print(obsnum)
                # N += J.transpose() @ P_obs @ J
                # b += J.transpose() @ P_obs @ l
                TotalObsNum += obsnum
            # N = B.transpose() @ R @ B
            # b = B.transpose() @ R @ L
            print(TotalObsNum / 3, "observations used," , StateLandmark / 3, "landmarks used")

            bPrior, NPrior_inv = self.CovConstraint_Kitti(windowsize_tmp, AllStateNum)
            # NPrior, bPrior, NPrior_inv = self.premarginalization_CLSTEST(windowsize_tmp, AllStateNum, StateFrame)
            # Ncom, bcom = self.compensateFEJ_CLSTEST(NPrior, windowsize)
            dx = self.compensateFEJ(windowsize)
            # print(np.all(np.abs(NPrior_inv - np.linalg.inv(NPrior)) < 1E-7))
            # np.savetxt("./log/NPrior.txt", NPrior)
            # np.savetxt("./log/NPrior_inv.txt", NPrior_inv)
            # NPrior_inv = np.linalg.inv(NPrior)
            # dx = NPrior_inv @ bcom
            # dx = self.compensateFEJ(windowsize_tmp)

            # if len(self.m_LandmarkLocal) or np.all(NPrior == 0):
            #     N[:6, :6] = N[:6, :6] + np.identity(6) * 1E7 # + np.identity(6) * 1E-7
            # else:
            #     np.savetxt("./debug/NPrior.txt", NPrior)
            #     np.savetxt("./debug/N.txt", N) + np.identity(N.shape[0]) * 1E-7
            # state = np.linalg.inv(N + NPrior) @ (b + bPrior + bcom)

            K = NPrior_inv @ B.transpose() @ np.linalg.inv(np.linalg.inv(R) + B @ NPrior_inv @ B.transpose())
            XPrior = NPrior_inv @ bPrior
            state = dx + XPrior + K @ (L - B @ (XPrior + dx))
            StateFrame = state[: windowsize_tmp * 6]

            for j in range(windowsize_tmp):  
                frames[j].m_pos =  frames[j].m_pos - state[j * 6: j * 6 + 3, :]
                frames[j].m_rota = frames[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
                frames[j].m_rota = UnitRotation(frames[j].m_rota)

            StateFrameNum = windowsize_tmp * 6
            for id_ in self.m_MapPoints.keys():
                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3, :]
            # self.check(map, camera)
            if self.removeOutlier(map, camera):
                self.loadstates(frames, self.m_MapPoints_Point)
                prevstate = state
                # # if iter > 0:
                # iter -= 1
                continue

            if iter >= 1 and np.linalg.norm(prevstate[: windowsize_tmp * 6] - state[: windowsize_tmp * 6], 2) < 1E-2:
                break
            # if iter != 9:
            #     prevstate = state
            iter += 1
        print(np.linalg.norm(prevstate[: windowsize_tmp * 6] - state[: windowsize_tmp * 6], 2))

        LocalFrame, i = {}, 0
        for frame in frames:
            LocalFrame[i] = frame
            i += 1
        if windowsize_tmp == windowsize:
            self.UpdateCov_Kitti(LocalFrame, NPrior_inv, camera, windowsize, bPrior)
            # self.marginalization_kitti(windowsize_tmp, LocalFrame, camera, NPrior, bPrior)
        count = 0
        for id, point in mappoints.items():
            if point.m_buse == -1:
                continue
            # if len(point.m_obs) > 4:
            #     point.m_buse = 1
                # continue
            # point.m_buse = True
            count += 1
            for obs in point.m_obs:
                obs.m_buse = True
        
        return 1
    

    def compensateFEJ_CLSTEST(self, NPrior, windowsize=20):
        FrameStateNum, StateLandmark = windowsize * 6, len(self.m_MapPoints) * 3
        Xdiff = np.zeros((FrameStateNum + StateLandmark, 1))
        Ncom, bcom = 0, np.zeros((FrameStateNum + StateLandmark, 1))
        if len(self.m_LandmarkLocal) == 0:
            return Ncom, bcom

        mapping = {}

        for mappointID, GlobalPos in self.m_MapPoints.items():
            if mappointID in self.m_LandmarkLocal.keys():
                LocalPos = self.m_LandmarkLocal[mappointID] * 3
                mapping[GlobalPos + FrameStateNum] = LocalPos
        if self.m_Nmarg.shape[0] != 0:
            # differences of constrained landmarks.
            # note that frames are not constrained in this case
            # frame of Xdiff remains 0
            Xdiff = np.zeros((FrameStateNum + StateLandmark, 1))
            for mappointID, point in self.m_MapPoints_Point.items():
                pos = self.m_MapPoints[mappointID] + FrameStateNum
                point_FEJ = self.getLandmarkFEJ(point)
                Xdiff[pos: pos + 3, :] = point.m_pos - point_FEJ
                # print(np.linalg.norm(Xdiff[pos: pos + 3, :]))

            
            Ncom, bcom = NPrior.copy(), NPrior @ Xdiff
            print(np.max(np.abs(Xdiff)), np.min(Xdiff))
            # J = np.linalg.cholesky(self.m_Nmarg).transpose()
            # J_return = np.zeros(NPrior.shape)
            # for gpos, lpos in mapping.items():
            #     for gpos1, lpos1 in mapping.items():
            #         J_return[gpos: gpos + 3, gpos1: gpos1 + 3] = J[lpos: lpos + 3, lpos1: lpos1 + 3]

            
            # validate / debug -----------
            # print (np.all(np.abs(J_return.transpose() @ J_return - NPrior)) < 1E-9)
            # marginalization should constrain landmarks only
            # print (np.all(NPrior[: FrameStateNum, : FrameStateNum] == 0))
            # print (np.all(Xdiff == 0))
            # --------------------
            
        return Ncom, bcom

    def marginalization_kitti(self, WindowSize, LocalFrame, camera, NPrior, bPrior):

        FirstLocalID = list(LocalFrame.keys())[0]
        frame = LocalFrame[FirstLocalID]
        tec, Rec = frame.m_pos, frame.m_rota
        features = frame.m_features

        # step 1: select to-be-removed states and measurements
        J, P_obs, l = self.setMEQ_Kitti(tec, Rec, features, camera, WindowSize, FirstLocalID)
        # P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        N = J.transpose() @ P_obs @ J
        HaveValue = np.diagonal(N) != 0
        N += NPrior

        N = N[[HaveValue[i] for i in range(len(HaveValue))], :]
        N_sub = N[:, [HaveValue[i] for i in range(len(HaveValue))]]

        b_all = J.transpose() @ P_obs @ l + bPrior
        b_sub = b_all[[HaveValue[i] for i in range(len(HaveValue))], :]
        
        # step 2: marginalization
        N11, N22, N12 = N_sub[: 6, : 6], N_sub[6:, 6: ], N_sub[: 6, 6: ]
        b1, b2 = b_sub[: 6, :], b_sub[6:, :]

        N12_T = N12.transpose()
        N11_inv = np.linalg.inv(N11)
        N_marg = N_sub # N22 - N12_T @ N11_inv @ N12
        b_marg = b_sub # b2 - N12_T @ N11_inv @ b1

        # step 3: specify map point ID -- position in N_marg
        items_to_remove = []
        for mappointID, value in self.m_LandmarkFEJ.items():
            if mappointID not in self.m_MapPoints.keys():
                items_to_remove.append(mappointID)
        
        for item in items_to_remove:
            del self.m_LandmarkFEJ[item]

        PosLandmarkStart = WindowSize * 6
        ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkLocal = {}
        for i in range(len(ConnectedNodes)):
            for mappointID, value in self.m_MapPoints.items():
                if ConnectedNodes[i] * 3 == value:
                    LandmarkLocal[mappointID] = i
                    # fix linearization point
                    if mappointID not in self.m_LandmarkFEJ.keys():
                        self.m_LandmarkFEJ[mappointID] = copy.deepcopy(self.m_MapPoints_Point[mappointID].m_pos)
                    break

        self.m_Nmarg = N_marg
        self.m_bmarg = b_marg
        self.m_LandmarkLocal = LandmarkLocal

    def removeOutlier(self, map, camera):
        frames = map.m_frames

        resi_dict = {}
        for frame in frames:
            tec, Rec = frame.m_pos, frame.m_rota
            features = frame.m_features
            for feat in features:
                mappoint = feat.m_mappoint
                if mappoint.m_buse < 1:
                    continue
                if feat.m_buse == False:
                    continue
                pointPos = mappoint.m_pos
                pointPos_c = np.matmul(Rec, (pointPos - tec))
                uv = camera.project(pointPos_c)
                resi_dict[feat] = np.linalg.norm(uv - feat.m_pos, 2)
        resi_dict = dict(sorted(resi_dict.items(), key=lambda x: x[1]))

        # chi2 test for outlier remove
        thres = 7.991
        iter, inlier, outlier = 0, 0, 0
        while iter < 5:
            for key, value in resi_dict.items():
                if value > thres:
                    outlier += 1
                else:
                    inlier += 1
            
            if float(inlier) / (outlier + inlier) > 0.5:
                break
            else:
                thres *= 2
                iter += 1
        
        bOutlier = False
        keys, values = list(resi_dict.keys()), list(resi_dict.values())
        for i in range(len(keys)):
            if resi_dict[keys[i]] >= thres:
                keys[i].m_buse = False
                if keys[i].m_frame.check() < 4:
                    for feat in keys[i].m_frame.m_features:
                        if resi_dict[keys[i]] >= 3 * thres:
                            feat.m_buse = False
                            continue
                        feat.m_buse = True
                    continue
                bOutlier = True
        



        # down = int(len(resi_dict) * 0.25)
        # up = int(len(resi_dict) * 0.75)
        
        # k = 1.5
        # normal_range = values[up] - values[down]
        # outlier_up = values[up] + k * normal_range

        # bOutlier = False
        # for i in range(up, len(keys)):
        #     if resi_dict[keys[i]] >= outlier_up:
        #         keys[i].m_buse = False
        #         bOutlier = True

        mappoints = map.m_points
        for id, point in mappoints.items():
            if point.m_buse == -1:
                continue
            
            for obs in point.m_obs:
                tc, Rc = obs.m_frame.m_pos, obs.m_frame.m_rota
                point_cam = Rc @ (point.m_pos - tc)
                if point_cam[2, 0] < 0:
                    point.m_buse = -1

            # point.check()

        for frame in frames:
            countl, countf = 0, 0
            for feat in frame.m_features:
                feat.m_mappoint.check()
                if feat.m_mappoint.m_buse == 1:
                    countl += 1
                if feat.m_buse:
                    countf += 1
            print("frame", frame.m_id, " used", countl, "landmarks,", countf, "features")
        return bOutlier

    def UpdateCov(self, LocalFrame, NPrior_inv, camera, windowsize, XPrior):
        FirstLocalID = list(LocalFrame.keys())[0]
        frame = LocalFrame[FirstLocalID]
        tec, Rec = frame.m_pos, frame.m_rota
        features = frame.m_features

        # step 1: update cov
        J, l = self.setMEQ_SW(tec, Rec, features, camera, windowsize, FirstLocalID)
        P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        HaveValue = [np.any(J[:, i] != 0) for i in range(J.shape[1])]
        N = J.transpose() @ P_obs @ J
        HaveValue = np.diagonal(N) != 0
        P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))

        J = J[:, [HaveValue[i] for i in range(len(HaveValue))]]
        # l = l[[HaveValue[i] for i in range(len(HaveValue))], :]
        NPrior = np.linalg.inv(NPrior_inv)
        bp = NPrior @ XPrior
        Np = NPrior[:, [HaveValue[i] for i in range(len(HaveValue))]]
        Np = Np[[HaveValue[i] for i in range(len(HaveValue))], :]
        # Xp = XPrior[[HaveValue[i] for i in range(len(HaveValue))], :]
        bp = bp[[HaveValue[i] for i in range(len(HaveValue))], :]

        self.m_Jmarg, self.m_lmarg, self.m_Npmarg, self.m_bpmarg = J, l, Np, bp
        # self.m_bpmarg = bp

        # step 2: specify map point ID -- position in N_marg
        items_to_remove = []
        for mappointID, value in self.m_LandmarkFEJ.items():
            if mappointID not in self.m_MapPoints.keys():
                items_to_remove.append(mappointID)
        
        for item in items_to_remove:
            del self.m_LandmarkFEJ[item]

        PosLandmarkStart = windowsize * 6
        ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkLocal = {}
        for i in range(len(ConnectedNodes)):
            for mappointID, value in self.m_MapPoints.items():
                if ConnectedNodes[i] * 3 == value:
                    LandmarkLocal[mappointID] = i
                    # fix linearization point
                    if mappointID not in self.m_LandmarkFEJ.keys():
                        self.m_LandmarkFEJ[mappointID] = copy.deepcopy(self.m_MapPoints_Point[mappointID].m_pos)
                    break

        self.m_LandmarkLocal = LandmarkLocal


    def UpdateCov_Kitti(self, LocalFrame, NPrior_inv, camera, windowsize, bPrior):
        FirstLocalID = list(LocalFrame.keys())[0]
        frame = LocalFrame[FirstLocalID]
        tec, Rec = frame.m_pos, frame.m_rota
        features = frame.m_features

        # step 1: update cov
        J, P_obs, l = self.setMEQ_Kitti(tec, Rec, features, camera, windowsize, FirstLocalID)
        # P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))
        HaveValue = [np.any(J[:, i] != 0) for i in range(J.shape[1])]
        # N = J.transpose() @ P_obs @ J
        # HaveValue = np.diagonal(N) != 0
        # P_obs = np.identity(J.shape[0]) * ( 1.0 / (self.m_PixelStd * self.m_PixelStd))

        J = J[:, [HaveValue[i] for i in range(len(HaveValue))]]
        # l = l[[HaveValue[i] for i in range(len(HaveValue))], :]
        NPrior = np.linalg.inv(NPrior_inv)
        Np = NPrior[:, [HaveValue[i] for i in range(len(HaveValue))]]
        Np = Np[[HaveValue[i] for i in range(len(HaveValue))], :]
        bp = bPrior[[HaveValue[i] for i in range(len(HaveValue))], :]

        self.m_Jmarg, self.m_lmarg, self.m_Npmarg, self.m_bpmarg = J, l, Np, bp
        self.m_Pobsmarg = P_obs

        # step 3: specify map point ID -- position in N_marg
        items_to_remove = []
        for mappointID, value in self.m_LandmarkFEJ.items():
            if mappointID not in self.m_MapPoints.keys():
                items_to_remove.append(mappointID)
        
        for item in items_to_remove:
            del self.m_LandmarkFEJ[item]

        PosLandmarkStart = windowsize * 6
        ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkLocal = {}
        for i in range(len(ConnectedNodes)):
            for mappointID, value in self.m_MapPoints.items():
                if ConnectedNodes[i] * 3 == value:
                    LandmarkLocal[mappointID] = i
                    # fix linearization point
                    if mappointID not in self.m_LandmarkFEJ.keys():
                        self.m_LandmarkFEJ[mappointID] = copy.deepcopy(self.m_MapPoints_Point[mappointID].m_pos)
                    break

        self.m_LandmarkLocal = LandmarkLocal


    def savestates(self, frames, mappoints):
        self.m_save_frames = copy.deepcopy(frames)
        self.m_save_mappoints = copy.deepcopy(mappoints)

    def loadstates(self, frames, mappoints):
        for i in range(len(frames)):
            frames[i].m_pos = self.m_save_frames[i].m_pos.copy()
            frames[i].m_rota = self.m_save_frames[i].m_rota.copy()
        
        for key, value in mappoints.items():
            value.m_pos = self.m_save_mappoints[key].m_pos.copy()

    def __addFeaturesKitti(self, features):
        count = 0
        for feat in features:
            mappoint = feat.m_mappoint
            mappoint.check()
            if mappoint.m_buse < 1:
                continue
            if feat.m_buse == True:
                count += 1
            mappointID = mappoint.m_id
            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
            self.m_MapPointPos += 3
        return count


    def setMEQ_Kitti(self, tec, Rec, features, camera, windowsize, LocalID):

        statenum = windowsize * 6 + len(self.m_MapPoints) * 3
        obsnum = 0
        FrameStateNum = windowsize * 6
        thres = 5.991

        for feat in features:
            mappoint = feat.m_mappoint
            if mappoint.m_buse < 1:
                continue
            if feat.m_buse == False:
                continue
            obsnum += 1

        # for row in range(len(features)):
        #     feat = features[row]
        #     mappoint = feat.m_mappoint
        #     if mappoint.m_buse >= 1:
        #         obsnum += 1
        obsnum *= 3
        J, l, P = np.zeros((obsnum, statenum)), np.zeros((obsnum, 1)), np.zeros((obsnum, obsnum))
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b

        row = 0
        for feat in features:
            mappoint = feat.m_mappoint
            if mappoint.m_buse < 1:
                continue
            if feat.m_buse == False:
                continue
            pointPos = self.getLandmarkFEJ(mappoint)
            pointID = mappoint.m_id
            nobs = len(mappoint.m_obs)
            a = np.linalg.norm(pointPos, 2)
            pointPos_c = np.matmul(Rec, (pointPos - tec))
            uv = camera.project(pointPos_c)
            uv_obs = feat.m_pos
            PointIndex = self.m_MapPoints[pointID]
            l_sub = uv - uv_obs
            l[row * 3: row * 3 + 3, :] = l_sub

            P_sub = robustKernelHuber(l_sub, thres)
            P_sub = np.diag(P_sub.flatten())
            P[row * 3: row * 3 + 3, row * 3: row * 3 + 3] = P_sub @ np.identity(3) * (1.0 / (self.m_PixelStd ** 2))

            Jphi = Jacobian_phai(Rec, tec, pointPos, pointPos_c, fx, fy, b)
            Jrcam = Jacobian_rcam(Rec, pointPos_c, fx, fy, b)
            JPoint = Jacobian_Point(Rec, pointPos_c, fx, fy, b)

            J[row * 3: row * 3 + 3, LocalID * 6 : LocalID * 6 + 3] = Jrcam
            J[row * 3: row * 3 + 3, LocalID * 6 + 3 : LocalID * 6 + 6] = Jphi
            J[row * 3: row * 3 + 3, FrameStateNum + PointIndex : FrameStateNum + PointIndex + 3] = JPoint
            row += 1

        return J, P, l

        