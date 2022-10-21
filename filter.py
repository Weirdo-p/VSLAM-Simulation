from operator import inv
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

        self.m_StateCov[:3, :3] *= PosStd
        self.m_StateCov[3:, 3:] *= AttStd

        self.m_MapPoints = {}       # position of mappoint in state vector
        self.m_MapPointPos = 6      # [cameraPos CameraRotation point1 ... pointN]

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
