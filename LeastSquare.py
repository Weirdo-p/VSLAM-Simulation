from multiprocessing import set_forkserver_preload
from turtle import position
import numpy as np
from vcommon import *
import copy

class CLS:
    def __init__(self, PixelStd=2, PointStd = 2, PosStd = 2, AttStd = 2 * D2R):
        self.m_PixelStd = PixelStd
        self.m_PointStd = PointStd
        self.m_PosStd = PosStd
        self.m_AttStd = AttStd
        self.m_MapPoints = {}       # position of mappoint in state vector
        self.m_MapPointPos = 0      # [cameraPos CameraRotation point1 ... pointN]
        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []

    def solveAll(self, frames, camera):
        obsnum, statenum = 0, 0
        print(id(frames))
        # count for observations and state
        count = 0
        for frame in frames:
            # if frame.m_time >= 10:
            #     break
            features = frame.m_features
            obsnum += len(features) * 3
            self.__addFeatures(features)
            self.m_estimateFrame.append(frame)

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        posStateNum = len(self.m_estimateFrame) * 6
        pointStateNum = len(self.m_MapPoints) * 3
        StateFrameNum = len(self.m_estimateFrame) * 6

        B, L = np.zeros((obsnum, statenum)), np.zeros((obsnum, 1))
        print(obsnum, statenum)
        
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b
        iter, MaxIter = 0, 10

        dXLast, P = 0, 0
        for iter in range(MaxIter):
            # form design matrix and residuals
            frame_i, obs_i = 0, 0
            for frame in self.m_estimateFrame:
                features = frame.m_features
                tec, Rec = frame.m_pos.copy(), frame.m_rota.copy()

                for row in range(len(features)):
                    feat = features[row]
                    mappoint = feat.m_mappoint
                    pointID = mappoint.m_id
                    pointPos = mappoint.m_pos
                    pointPos_c = np.matmul(Rec, (pointPos - tec))
                    uv = camera.project(pointPos_c)
                    uv_obs = feat.m_pos
                    l_sub = uv - uv_obs

                    Jphi = Jacobian_phai(Rec, tec, pointPos, pointPos_c, fx, fy, b)
                    Jrcam = Jacobian_rcam(Rec, pointPos_c, fx, fy, b)
                    JPoint = Jacobian_Point(Rec, pointPos_c, fx, fy, b)
                    PointIndex = self.m_MapPoints[pointID]


                    L[obs_i * 3: (obs_i + 1) * 3, :] = l_sub

                    B[obs_i * 3: (obs_i + 1) * 3, frame_i: frame_i + 3] = Jrcam
                    B[obs_i * 3: (obs_i + 1) * 3, frame_i + 3 : frame_i + 6] = Jphi
                    B[obs_i * 3: (obs_i + 1) * 3, StateFrameNum + PointIndex: StateFrameNum + PointIndex + 3] = JPoint
                    obs_i += 1

                frame_i += 6
            
            B_all, L_all = np.zeros((obsnum + statenum, statenum)), np.zeros((obsnum + statenum, 1))
            B_all[:statenum, :statenum] = np.identity(statenum)
            B_all[statenum:, :] = B

            L_all[: statenum, :] = np.zeros((statenum, 1))
            L_all[statenum:, :] = L

            P = np.identity(obsnum + statenum)
            row_P = 0
            p = np.identity(6)
            p[:3, :3] *= self.m_PosStd * self.m_PosStd
            p[3:6, 3:6] *= self.m_AttStd * self.m_AttStd
            while True:
                P[row_P:row_P + 6, row_P:row_P + 6] = p
                row_P += 6
                if row_P >= P.shape[0]-6:
                    break

            P[posStateNum:posStateNum + pointStateNum, posStateNum:posStateNum + pointStateNum] = np.identity(pointStateNum) * self.m_PointStd * self.m_PointStd
            P[statenum:, statenum:] = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd   
            P = np.linalg.inv(P)
            print("calculate dx")

            dx = np.linalg.inv(B_all.transpose() @ P @ B_all) @ B_all.transpose() @ P @ L_all
            print("one loop done")

            for i in range(len(self.m_estimateFrame)):
                frameDX = dx[i * 6: (i + 1) * 6, :]
                self.m_estimateFrame[i].m_pos = self.m_estimateFrame[i].m_pos - frameDX[0: 3]
                self.m_estimateFrame[i].m_rota = self.m_estimateFrame[i].m_rota @ (np.identity(3) - SkewSymmetricMatrix(frameDX[3: 6]))
            
                
            for id_ in self.m_MapPoints.keys():

                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= dx[StateFrameNum + position : StateFrameNum + position + 3]


            if iter != 0:
                print("test", np.linalg.norm(dx[:posStateNum, :] - dXLast[:posStateNum, :], 2))
                print(np.max(np.abs(dx[:posStateNum, :] - dXLast[:posStateNum, :])))


            if iter != 0 and (np.linalg.norm(dx[:posStateNum] - dXLast[:posStateNum], 2) < 1e-2 or np.max(np.abs(dx[:posStateNum, :] - dXLast[:posStateNum, :])) < 5E-2):
                break
            
            dXLast = copy.deepcopy(dx)
        
        return self.m_estimateFrame

    def __addFeatures(self, features):
        for feat in features:
            mappoint = feat.m_mappoint
            mappointID = mappoint.m_id
            if mappointID in self.m_MapPoints.keys():
                continue
            self.m_MapPoints[mappointID] = self.m_MapPointPos
            self.m_MapPoints_Point[mappointID] = mappoint
            self.m_MapPointPos += 3
