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

    def solveAll(self, frames, camera, frames_gt, maxtime = -1):
        obsnum, statenum = 0, 0
        # count for observations and state
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time

        for frame in frames:
            if frame.m_time > LastTime:
                break
            features = frame.m_features
            obsnum += len(features) * 3
            self.__addFeatures(features)
            self.m_estimateFrame.append(frame)

        statenum = len(self.m_estimateFrame) * 6 + len(self.m_MapPoints) * 3
        posStateNum = len(self.m_estimateFrame) * 6
        pointStateNum = len(self.m_MapPoints) * 3
        StateFrameNum = len(self.m_estimateFrame) * 6

        B, L = np.zeros((obsnum, statenum)), np.zeros((obsnum, 1))
        if statenum >= obsnum:
            print("obs num not enough")
            exit(-1)
        # print(obsnum, statenum)
        
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b
        iter, MaxIter = 0, len(self.m_estimateFrame)

        dXLast, P = 0, 0
        #%%normal
        # P = np.identity(obsnum + statenum)
        # row_P = 0
        # p = np.identity(6)
        # p[:3, :3] *= (self.m_PosStd * self.m_PosStd)
        # p[3:6, 3:6] *= (self.m_AttStd * self.m_AttStd)
        # while True:
        #     P[row_P:row_P + 6, row_P:row_P + 6] = p
        #     row_P += 6
        #     if row_P >= P.shape[0] - 6:
        #         break
        # P[posStateNum:posStateNum + pointStateNum, posStateNum:posStateNum + pointStateNum] = np.identity(pointStateNum) * self.m_PointStd * self.m_PointStd
        # P[posStateNum + pointStateNum:, posStateNum + pointStateNum: ] = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
        # P = np.linalg.inv(P)
        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov_CLS.txt", np.linalg.inv(P))
        #%% test
        P = np.identity(obsnum) * (self.m_PixelStd ** 2)
        P_Prior = np.identity(statenum)
        row_P = 0
        p = np.identity(6)
        p[:3, :3] *= (self.m_PosStd * self.m_PosStd)
        p[3:6, 3:6] *= (self.m_AttStd * self.m_AttStd)
        while True:
            P_Prior[row_P:row_P + 6, row_P:row_P + 6] = p
            row_P += 6
            if row_P >= P_Prior.shape[0] - 6:
                break
        
        P_Prior[posStateNum:posStateNum + pointStateNum, posStateNum:posStateNum + pointStateNum] = np.identity(pointStateNum) * self.m_PointStd * self.m_PointStd
        # P = np.linalg.inv(P)
        P_Prior = np.linalg.inv(P_Prior)
        N, W = np.zeros((statenum, statenum)), np.zeros((statenum, 1))

        
        # print(self.m_AttStd)
        for iter in range(MaxIter):
            print(iter)
            # form design matrix and residuals
            frame_i, obs_i, i = 0, 0, 0
            for i in range(len(self.m_estimateFrame)):
                features = frames_gt[i].m_features
                tec, Rec = frames_gt[i].m_pos.copy(), frames_gt[i].m_rota.copy()
                B_part, L_part = np.zeros((len(features) * 3, statenum)), np.zeros((len(features) * 3, 1))
                P_part = np.identity(len(features) * 3) * self.m_PixelStd * self.m_PixelStd
                P_part = np.linalg.inv(P_part)
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

                    #%% normal
                    # L[obs_i * 3: (obs_i + 1) * 3, :] = l_sub

                    # B[obs_i * 3: (obs_i + 1) * 3, frame_i: frame_i + 3] = Jrcam
                    # B[obs_i * 3: (obs_i + 1) * 3, frame_i + 3 : frame_i + 6] = Jphi
                    # B[obs_i * 3: (obs_i + 1) * 3, StateFrameNum + PointIndex: StateFrameNum + PointIndex + 3] = JPoint

                    #%% test
                    L_part[row * 3: (row + 1) * 3, :] = l_sub

                    B_part[row * 3: (row + 1) * 3, frame_i: frame_i + 3] = Jrcam
                    B_part[row * 3: (row + 1) * 3, frame_i + 3 : frame_i + 6] = Jphi
                    B_part[row * 3: (row + 1) * 3, StateFrameNum + PointIndex: StateFrameNum + PointIndex + 3] = JPoint
                    obs_i += 1
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_CLS_" + str(i) + ".txt", B_part)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_CLS_" + str(i) + ".txt", L_part)
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/W_CLS_" + str(i) + ".txt", P_part)
                N += B_part.transpose() @ P_part @ B_part
                W += B_part.transpose() @ P_part @ L_part
                frame_i += 6
                i += 1
            #%% normal
            # B_all, L_all = np.zeros((obsnum + statenum, statenum)), np.zeros((obsnum + statenum, 1))
            # B_all[:statenum, :statenum] = np.identity(statenum)
            # B_all[statenum:, :] = B

            # L_all[: statenum, :] = np.zeros((statenum, 1))
            # L_all[statenum:, :] = L

            # print("calculate dx")
            # # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_CLS.txt", B_all)
            # # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_CLS.txt", L_all)
            # # break
            # P[statenum:, statenum:] = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd   
            # dx = np.linalg.inv(B_all.transpose() @ P @ B_all) @ B_all.transpose() @ P @ L_all
            #%% test
            B_prior = np.identity(statenum)
            N_prior = B_prior.transpose() @ P_Prior @ B_prior

            print("calculate dx")
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_CLS.txt", B_all)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_CLS.txt", L_all)
            # break
            # P[statenum:, statenum:] = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd   
            dx = np.linalg.inv(N + N_prior) @ W 
            print("one loop done")

            # print(dx)
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


            if iter != 0 and (np.linalg.norm(dx[:posStateNum] - dXLast[:posStateNum], 2) < 1e-2 or np.max(np.abs(dx[:posStateNum, :] - dXLast[:posStateNum, :])) < 1E-2):
                break
            
            dXLast = copy.deepcopy(dx)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/P.txt", np.linalg.inv(P))
            # P = np.linalg.inv(P)
            # P[:statenum, :statenum] = np.linalg.inv(B_all.transpose() @ P @ B_all)
            # break
        
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
    

    def solveSequential(self, frames, camera, maxtime=-1):
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
        self.m_StateCov = np.linalg.inv(self.m_StateCov)
        
        for i in range(len(self.m_estimateFrame)):
            frame = self.m_estimateFrame[i]
            tec, Rec = frame.m_pos, frame.m_rota
            features = frame.m_features
            obsnum = len(features) * 3
            R = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd
            J, l = self.setMEQ_AllState(tec, Rec, features, camera, i)
            W = np.linalg.inv(R)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_FILTER_" + str(i) + ".txt", J)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_FILTER_" + str(i) + ".txt", l)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/W_FILTER_" + str(i) + ".txt", W)
            print("Process " + str(i) + "th frame")
            N = J.transpose() @ W @ J + self.m_StateCov
            w = J.transpose() @ W @ l
            state = np.linalg.inv(N) @ w

            self.m_StateCov = N
            # update current frame
            FramedX = state[i * 6: i * 6 + 6, :]
            self.m_estimateFrame[i].m_pos = self.m_estimateFrame[i].m_pos - FramedX[0: 3]
            self.m_estimateFrame[i].m_rota = self.m_estimateFrame[i].m_rota @ (np.identity(3) - SkewSymmetricMatrix(FramedX[3: 6]))

            # for feat in features:
            #     MapPointID = feat.m_mappoint.m_id
            #     MapPointPos = self.m_MapPoints[MapPointID]
            #     self.m_MapPoints_Point[MapPointID].m_pos = self.m_MapPoints_Point[MapPointID].m_pos - state[StateFrameNum + MapPointPos: StateFrameNum + MapPointPos + 3, :]

            # update all frames
            # for j in range(len(self.m_estimateFrame)):  
            #     self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
            #     self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

            # for id_ in self.m_MapPoints.keys():

            #     position = self.m_MapPoints[id_]
            #     self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]
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
