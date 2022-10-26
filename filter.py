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

    def filter_AllState(self, frames, camera, maxtime=-1):
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
        Q = np.identity(statenum)       # noise during state transition
        # Q[:StateFrameNum, : StateFrameNum] = self.m_QPose
        i = 0
        while True:
            Q[i : i + 6, i : i + 6] = self.m_QPose
            i += 6
            if i >= statenum - 6:
                break

        Q[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * self.m_QPoint
        print(Q)
        self.m_StateCov = np.identity(statenum)
        PoseCov = np.identity(6)
        PoseCov[:3, :3] = self.m_PosStd ** 2
        PoseCov[3:, 3:] = self.m_AttStd ** 2

        print(self.m_QPoint, self.m_PosStd, self.m_AttStd * R2D)
        i = 0
        while True:
            self.m_StateCov[i: i + 6, i: i + 6] = PoseCov
            i += 6

            if i >= self.m_StateCov.shape[0] - 6:
                break
        self.m_StateCov[StateFrameNum:, StateFrameNum:] = np.identity(pointStateNum) * (self.m_PosStd ** 2)

        for i in range(len(self.m_estimateFrame)):
            frame = self.m_estimateFrame[i]
            tec, Rec = frame.m_pos, frame.m_rota
            features = frame.m_features
            obsnum = len(features) * 3
            R = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
            J, l = self.setMEQ_AllState(tec, Rec, features, camera, i)
            print("Process " + str(i) + "th frame")
            state_cov_pre = Phi @ self.m_StateCov @ Phi.transpose() + Q
            K = state_cov_pre @ J.transpose() @ np.linalg.inv(J @ state_cov_pre @ J.transpose() + R)
            state = K @ l
            tmp = (np.identity(K.shape[0]) - K @ J)
            self.m_StateCov = tmp @ state_cov_pre @ tmp.transpose() + K @ R @ K.transpose()

            # update current frame
            FramedX = state[i * 6: i * 6 + 6, :]
            self.m_estimateFrame[i].m_pos = self.m_estimateFrame[i].m_pos - FramedX[0: 3]
            self.m_estimateFrame[i].m_rota = self.m_estimateFrame[i].m_rota @ (np.identity(3) + SkewSymmetricMatrix(FramedX[3: 6]))

            # for feat in features:
            #     MapPointID = feat.m_mappoint.m_id
            #     MapPointPos = self.m_MapPoints[MapPointID]
            #     self.m_MapPoints_Point[MapPointID].m_pos = self.m_MapPoints_Point[MapPointID].m_pos - state[StateFrameNum + MapPointPos: StateFrameNum + MapPointPos + 3, :]

            # update all frames
            # for j in range(len(self.m_estimateFrame)):  
            #     self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
            #     self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

            for id_ in self.m_MapPoints.keys():

                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]
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

    def __addFeatures(self, features):
        for feat in features:
            mappoint = feat.m_mappoint
            mappointID = mappoint.m_id
            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
            self.m_MapPointPos += 3
