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

        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []

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

        # ????????? ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????

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

    def filter_AllState(self, frames, camera, frames_gt, maxtime=-1, iteration = 1):
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
            for LocalID, frame in LocalFrames_gt.items():
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


    def __addFeatures(self, features):
        for feat in features:
            mappoint = feat.m_mappoint
            mappointID = mappoint.m_id
            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
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

            J[row * 3: row * 3 + 3, LocalID * 6 : LocalID * 6 + 3] = Jrcam
            J[row * 3: row * 3 + 3, LocalID * 6 + 3 : LocalID * 6 + 6] = Jphi
            J[row * 3: row * 3 + 3, FrameStateNum + PointIndex : FrameStateNum + PointIndex + 3] = JPoint

        return J, l
