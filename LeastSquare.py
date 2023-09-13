from multiprocessing import set_forkserver_preload
from turtle import position
import numpy as np
from vcommon import *
import cv2 as cv

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
        self.m_Nmarg = 0
        self.m_bmarg = 0
        self.m_LandmarkLocal = {}
        self.m_FrameFEJ = {}
        self.m_LandmarkFEJ = {}
    
    def reset(self):
        self.m_MapPoints = {}       # position of mappoint in state vector
        self.m_MapPointPos = 0      # [cameraPos CameraRotation point1 ... pointN]
        self.m_MapPoints_Point = {}
        self.m_estimateFrame = []
        self.m_Nmarg = 0
        self.m_bmarg = 0
        self.m_LandmarkLocal = {}

    def solveAll(self, frames, camera, frames_gt, maxtime = -1, iteration = 1):
        obsnum, statenum = 0, 0
        # count for observations and state
        LastTime = maxtime
        if maxtime > frames[len(frames) - 1].m_time or maxtime == -1:
            LastTime = frames[len(frames) - 1].m_time

        for frame in frames:
            if frame.m_time > LastTime:
                break
            print("add " + str(frame.m_id) + "th frame")
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
        print(obsnum, statenum)
        
        fx, fy, b = camera.m_fx, camera.m_fy, camera.m_b
        iter, MaxIter = 0, iteration

        dXLast, P = 0, 0
        P = np.identity(obsnum + statenum)
        row_P = 0
        p = np.identity(6)
        p[:3, :3] *= (self.m_PosStd * self.m_PosStd)
        p[3:6, 3:6] *= (self.m_AttStd * self.m_AttStd)
        while True:
            P[row_P:row_P + 6, row_P:row_P + 6] = p
            row_P += 6
            if row_P >= P.shape[0] - 6:
                break
        P[posStateNum:posStateNum + pointStateNum, posStateNum:posStateNum + pointStateNum] = np.identity(pointStateNum) * self.m_PointStd * self.m_PointStd
        # P = np.linalg.inv(P)

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

            print("calculate dx")
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/H_CLS.txt", B_all)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L_CLS.txt", L_all)
            # break
            P[statenum:, statenum:] = np.identity(obsnum) * self.m_PixelStd * self.m_PixelStd   
            P = np.linalg.inv(P)
            dx = np.linalg.inv(B_all.transpose() @ P @ B_all) @ B_all.transpose() @ P @ L_all
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/B.txt", B_all)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/P.txt", P)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/L.txt", L_all)
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
            break
        
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


    def solveSequential(self, frames, camera, maxtime=-1, iteration=1):
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

        MatIteration = iteration
        if iteration == -1:
            MatIteration = 10
        
        dXLast = 0
        for iter in range(MatIteration):
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
            # if iter == 1:
            #     np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/CovCLS.txt", self.m_StateCov)
            #     exit(-1)
            N, w = 0, 0
            state = np.zeros((statenum, 1))
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
                w = J.transpose() @ W @ l + self.m_StateCov @ state

                self.m_StateCov = N
                # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/COV_CLS.txt", np.linalg.inv(self.m_StateCov))
                state = np.linalg.inv(N ) @ w
                # break
            # update current frame
            # FramedX = state[i * 6: i * 6 + 6, :]
            # self.m_estimateFrame[i].m_pos = self.m_estimateFrame[i].m_pos - FramedX[0: 3]
            # self.m_estimateFrame[i].m_rota = self.m_estimateFrame[i].m_rota @ (np.identity(3) - SkewSymmetricMatrix(FramedX[3: 6]))

            # for feat in features:
            #     MapPointID = feat.m_mappoint.m_id
            #     MapPointPos = self.m_MapPoints[MapPointID]
            #     self.m_MapPoints_Point[MapPointID].m_pos = self.m_MapPoints_Point[MapPointID].m_pos - state[StateFrameNum + MapPointPos: StateFrameNum + MapPointPos + 3, :]
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/dx.txt", state)
            # update all frames
            for j in range(len(self.m_estimateFrame)):  
                self.m_estimateFrame[j].m_pos = self.m_estimateFrame[j].m_pos - state[j * 6: j * 6 + 3, :]
                self.m_estimateFrame[j].m_rota = self.m_estimateFrame[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

            for id_ in self.m_MapPoints.keys():

                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]
            
            if iteration != -1:
                continue

            if iter != 0:
                print("test", np.linalg.norm(state[:StateFrameNum, :] - dXLast[:StateFrameNum, :], 2))
                print(np.max(np.abs(state[:StateFrameNum, :] - dXLast[:StateFrameNum, :])))


            if iter != 0 and (np.linalg.norm(state[:StateFrameNum] - dXLast[:StateFrameNum], 2) < 1e-2 or np.max(np.abs(state[:StateFrameNum, :] - dXLast[:StateFrameNum, :])) < 1E-2):
                break
            
            dXLast = state.copy()
            print(iter)
        return frames
            

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
        for i in range(len(frames)):
            if frames[i].m_time > LastTime:
                break
            LocalFrames[Local] = frames[i]
            LocalFrames_gt[Local] = frames_gt[i]

            Local += 1
            if Local < windowsize:
                continue
            tmp = (windowsize - 1) * 6
            self.m_StateCov[tmp: tmp + 6 , tmp: tmp + 6] = PoseCov
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov_prior.txt", self.m_StateCov)
            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId, frame in LocalFrames.items():
                nobs += len(frame.m_features) * 3
                self.__addFeatures(frame.m_features)
            StateLandmark = len(self.m_MapPoints) * 3
            print("process " + str(frame.m_id) + "th frame. Landmark: " + str(len(self.m_MapPoints)) + ", observation num: " + str(nobs / 3) + ", Local frame size: " + str(len(LocalFrames)))
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

            self.m_StateCov = np.zeros((StateFrameSize + StateLandmark, StateFrameSize + StateLandmark))

            if len(self.m_LandmarkLocal) == 0:
                j = 0
                while True:
                    self.m_StateCov[j: j + 6, j: j + 6] = PoseCov
                    j += 6

                    if j >= StateFrameSize:
                        break
                self.m_StateCov[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) *  self.m_PointStd * self.m_PointStd
                tmp = (windowsize - 1) * 6
                self.m_StateCov = np.linalg.inv(self.m_StateCov)
            
            StateFrame = np.zeros((AllStateNum, 1))
            self.m_StateCov[tmp: StateFrameSize, tmp:StateFrameSize] = np.linalg.inv(PoseCov)

            self.PassCov(False, windowsize)
            B_all, L_all = np.zeros((nobs + AllStateNum, AllStateNum)), np.zeros((nobs + AllStateNum, 1))
            P_all = np.zeros((AllStateNum + nobs, AllStateNum + nobs))

            # prior part
            B_all[: AllStateNum, :AllStateNum] = np.identity(AllStateNum)
            P_all[: AllStateNum, :AllStateNum] = self.m_StateCov
            L_all[: AllStateNum, :] = StateFrame 

            # observation part
            B_all[AllStateNum:, ] = B
            P_all[AllStateNum:, AllStateNum: ] = np.identity(nobs) * (1.0 / self.m_PixelStd / self.m_PixelStd)
            L_all[AllStateNum:, ] = L
            # P_all = np.linalg.inv(P_all)

            N = B_all.transpose() @ P_all @ B_all
            b = B_all.transpose() @ P_all @ L_all
            state = np.linalg.inv(N) @ b
            StateFrame = state[: windowsize * 6]
            self.m_StateCov = N #[: windowsize * 6, : windowsize * 6]
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov.txt", state)


            # 2. update states. evaluate jacobian at groundtruth, do not update.
            for j in range(Local):  
                LocalFrames[j].m_pos = LocalFrames[j].m_pos - state[j * 6: j * 6 + 3, :]
                LocalFrames[j].m_rota = LocalFrames[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))

            StateFrameNum = windowsize * 6
            for id_ in self.m_MapPoints.keys():
                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3]

            # 3. remove old frame and its covariance
            for _id in range(Local - 1):
                LocalFrames_gt[_id] = LocalFrames_gt[_id + 1]
                LocalFrames[_id] = LocalFrames[_id + 1]
            tmp = (windowsize - 1) * 6
            self.m_StateCov[: tmp, : tmp] = self.m_StateCov[6: StateFrameSize, 6: StateFrameSize]
            self.m_StateCov[tmp: tmp + 6, :] = 0
            self.m_StateCov[:, tmp: tmp + 6] = 0
            Local -= 1
            StateFrame[: tmp, :] = StateFrame[6:, :]
            StateFrame[tmp:, :] = 0

            self.PassCov(True)
        return frames

    def PassCov(self, bPost, windowsize = 20):
        if bPost:
            for mappointID, value in self.m_MapPoints.items():
                self.m_LandmarkLocal[mappointID] = value
            return
        
        if len(self.m_LandmarkLocal) == 0:
            return

        NewCovSize = windowsize * 6 + len(self.m_MapPoints) * 3
        cov = np.zeros((NewCovSize, NewCovSize))

        PoseNum = windowsize * 6
        cov[:PoseNum, :PoseNum] = self.m_StateCov[:PoseNum, :PoseNum]

        PosMap = {}
        for i in range(windowsize):
            PosMap[i * 3] = i * 3

        for id_old, pos_old in self.m_LandmarkLocal.items():
            if id_old in self.m_LandmarkLocal.keys():
                PosMap[pos_old + PoseNum] = self.m_LandmarkLocal[id_old] + PoseNum

        for old_pos_row, new_pos_row in PosMap.items():
            for old_pos_col, new_pos_col in PosMap.items():
                cov[new_pos_row: new_pos_row + 3, new_pos_col: new_pos_col + 3] = self.m_StateCov[old_pos_row: old_pos_row + 3, old_pos_col: old_pos_col + 3]

        self.m_StateCov = cov

        

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
        for i in range(len(frames)):
            if frames[i].m_time > LastTime:
                break
            LocalFrames[Local] = frames[i]
            LocalFrames_gt[Local] = frames_gt[i]

            Local += 1
            if Local < windowsize:
                continue
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/Cov_prior.txt", self.m_StateCov)
            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId, frame in LocalFrames.items():
                nobs += len(frame.m_features) * 3
                self.__addFeatures(frame.m_features)
            StateLandmark = len(self.m_MapPoints) * 3

            self.m_StateCov = np.zeros((StateFrameSize + StateLandmark, StateFrameSize + StateLandmark))
            j = 0
            while True:
                self.m_StateCov[j: j + 6, j: j + 6] = PoseCov
                j += 6

                if j >= StateFrameSize:
                    break
            self.m_StateCov[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
            tmp = (windowsize - 1) * 6
            self.m_StateCov[tmp: StateFrameSize, tmp:StateFrameSize] = PoseCov
            
            print("process " + str(frame.m_id) + "th frame. Landmark: " + str(len(self.m_MapPoints)) + ", observation num: " + str(nobs / 3) + ", Local frame size: " + str(len(LocalFrames)))
            #TODO: 1. solve CLS problem by marginalizing landmark
            AllStateNum = windowsize * 6 + StateLandmark
            TotalObsNum = 0
            B, L = np.zeros((nobs, AllStateNum)), np.zeros((nobs, 1))
            for LocalID, frame in LocalFrames.items():
                # frame_FEJ = self.getFrameFEJ(frame)
                tec, Rec = frame.m_pos, frame.m_rota
                features = frame.m_features
                obsnum = len(features) * 3
                J, l = self.setMEQ_SW(tec, Rec, features, camera, windowsize, LocalID)

                B[TotalObsNum : TotalObsNum + obsnum, :] = J
                L[TotalObsNum : TotalObsNum + obsnum, :] = l
                TotalObsNum += obsnum

            NPrior, bPrior = self.premarginalization(windowsize, AllStateNum, StateFrame)
            Ncom, bcom = self.compensateFEJ(NPrior, windowsize)
            # print (np.max(bcom))

            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/NPrior.txt", NPrior)
            # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/bPrior.txt", bPrior)
            P_obs = np.identity(nobs) * (1.0 / (self.m_PixelStd * self.m_PixelStd))

            N = B.transpose() @ P_obs @ B + NPrior
            b = B.transpose() @ P_obs @ L + bPrior + bcom
            state = np.linalg.inv(N) @ b
            StateFrame = state[: windowsize * 6]
            self.m_StateCov = np.linalg.inv(N)[: windowsize * 6, : windowsize * 6]

            self.marginalization(windowsize, LocalFrames, camera, NPrior, bPrior)
            # 2. update states. evaluate jacobian at groundtruth, do not update.
            for j in range(Local):  
                LocalFrames[j].m_pos = LocalFrames[j].m_pos - state[j * 6: j * 6 + 3, :]
                LocalFrames[j].m_rota = LocalFrames[j].m_rota @ (np.identity(3) - SkewSymmetricMatrix(state[j * 6 + 3: j * 6 + 6, :]))
            StateFrameNum = windowsize * 6

            for id_ in self.m_MapPoints.keys():
                position = self.m_MapPoints[id_]
                self.m_MapPoints_Point[id_].m_pos -= state[StateFrameNum + position : StateFrameNum + position + 3, :]

            # 3. remove old frame and its covariance
            # TODO: marginalization should be applied
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
        return frames

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
        while iter < 20:
            # 1. search for observations and landmarks
            self.m_MapPoints = {}
            self.m_MapPoints_Point = {}
            self.m_MapPointPos = 0
            nobs = 0
            for LocalId in range(len(frames)):
                frame = frames[LocalId]
                nobs += len(frame.m_features) * 3
                self.__addFeaturesKitti(frame.m_features)
            StateLandmark = len(self.m_MapPoints) * 3
            self.savestates(frames, self.m_MapPoints_Point)

            if len(self.m_LandmarkLocal) == 0:
                self.m_StateCov = np.zeros((StateFrameSize + StateLandmark, StateFrameSize + StateLandmark))
                j = 0
                while True:
                    self.m_StateCov[j: j + 6, j: j + 6] = PoseCov
                    j += 6

                    if j >= StateFrameSize:
                        break
                self.m_StateCov[StateFrameSize:, StateFrameSize: ] = np.identity(StateLandmark) * self.m_PointStd * self.m_PointStd
                tmp = (windowsize_tmp - 1) * 6
                self.m_StateCov[tmp: StateFrameSize, tmp:StateFrameSize] = PoseCov
            # 1. solve CLS problem by marginalizing landmark
            AllStateNum = windowsize_tmp * 6 + StateLandmark
            TotalObsNum = 0
            R = np.zeros((nobs, nobs))
            B, L = np.zeros((nobs, AllStateNum)), np.zeros((nobs, 1))
            # N, b = np.zeros((AllStateNum, AllStateNum)), np.zeros((AllStateNum, 1))
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
            N = B.transpose() @ R @ B
            b = B.transpose() @ R @ L
            print(TotalObsNum / 3, "observations used," , StateLandmark / 3, "landmarks used")
            NPrior, bPrior = self.premarginalization(windowsize_tmp, AllStateNum, StateFrame)

            if len(self.m_LandmarkLocal) or np.all(NPrior == 0):
                N[:6, :6] = N[:6, :6] + np.identity(6) * 1E7 # + np.identity(6) * 1E-7
            # else:
            #     np.savetxt("./debug/NPrior.txt", NPrior)
            #     np.savetxt("./debug/N.txt", N) + np.identity(N.shape[0]) * 1E-7
            state = np.linalg.inv(N + NPrior + np.identity(N.shape[0]) * 1E-8) @ (b + bPrior)
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
                # if iter > 0:
                iter -= 1
                continue

            if iter >= 1 and np.linalg.norm(prevstate[: windowsize_tmp * 6] - state[: windowsize_tmp * 6], 2) < 1E-2:
                break
            if iter != 9:
                prevstate = state
            iter += 1
        print(np.linalg.norm(prevstate[: windowsize_tmp * 6] - state[: windowsize_tmp * 6], 2))
        if windowsize_tmp == windowsize:
            self.marginalization(N, b, windowsize_tmp)
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

    def savestates(self, frames, mappoints):
        self.m_save_frames = copy.deepcopy(frames)
        self.m_save_mappoints = copy.deepcopy(mappoints)

    def loadstates(self, frames, mappoints):
        for i in range(len(frames)):
            frames[i].m_pos = self.m_save_frames[i].m_pos.copy()
            frames[i].m_rota = self.m_save_frames[i].m_rota.copy()
        
        for key, value in mappoints.items():
            value.m_pos = self.m_save_mappoints[key].m_pos.copy()
            

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

    def check(self, map, camera):
        mappoints = map.m_points
        for id, point in mappoints.items():
            if point.m_buse == -1:
                continue
            
            for obs in point.m_obs:
                tc, Rc = obs.m_frame.m_pos, obs.m_frame.m_rota
                point_cam = Rc @ (point.m_pos - tc)
                if point_cam[2, 0] < 0 or point_cam[2, 0] >= 200:
                    point.m_buse = -1
                
                resi = camera.project(point_cam) - obs.m_pos
                if np.abs(resi[0]) > 10 or np.abs(resi[1])> 10:
                    obs.m_buse = 0

            count = 0
            for obs in point.m_obs:
                if obs.m_buse == False:
                    count += 1
            
            if count == len(point.m_obs):
                point.m_buse = 0
            
            
    
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

        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/N_sub.txt", N_sub)
        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/N.txt", N)
        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/b_sub.txt", b_sub)
        # np.savetxt("/home/xuzhuo/Documents/code/python/01-master/visual_simulation/log/debug/b.txt", b)

    def getFrameFEJ(self, frame):
        id_to_find = frame.m_id
        if id_to_find in self.m_FrameFEJ.keys():
            return self.m_FrameFEJ[id_to_find]
        else:
            frame_to_return = copy.deepcopy(frame)
            self.m_FrameFEJ[id_to_find] = frame_to_return
            return frame_to_return

    def getLandmarkFEJ(self, landmark):
        id_to_find = landmark.m_id
        if id_to_find in self.m_LandmarkFEJ.keys():
            return self.m_LandmarkFEJ[id_to_find]
        else:
            # landmark_to_return = copy.deepcopy(landmark)
            # self.m_LandmarkFEJ[id_to_find] = landmark_to_return
            return landmark.m_pos.copy()

    def compensateFEJ(self, NPrior, windowsize=20):
        Ncom, bcom = 0, 0
        if len(self.m_LandmarkLocal) == 0:
            return Ncom, bcom

        FrameStateNum, StateLandmark = windowsize * 6, len(self.m_MapPoints) * 3
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
            Nmarg, bmarg = self.submarg()
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
                    # NPrior_inv[gpos: gpos + 3, gpos1: gpos1 + 3] = self.m_Dmarg[lpos: lpos + 3, lpos1: lpos1 + 3]
                bPrior[gpos: gpos + 3, : ] = bmarg[lpos: lpos + 3, :]
            
        return NPrior, bPrior #, NPrior_inv, X_return, dX
        
    def submarg(self):
        N_sub, b_sub = self.m_Nmarg, self.m_bmarg
        # check observability of landmarks
        # ConnectedNodes = [int((index - PosLandmarkStart) / 3)  for index in range(len(HaveValue)) if HaveValue[index] and index >=120 and index % 3 == 0]
        LandmarkRemove = []
       
        for mappointid, localpos in self.m_LandmarkLocal.items():
            if mappointid not in self.m_MapPoints.keys():
                for i in range(3):
                    LandmarkRemove.append((mappointid, localpos + i + 6))
        
        for index in LandmarkRemove:
            id = index[0]
            N_sub = np.delete(N_sub, index[1], 0)
            N_sub = np.delete(N_sub, index[1], 1)
            b_sub = np.delete(b_sub, index[1], 0)

        for index in LandmarkRemove:
            if index[0] in self.m_LandmarkLocal.keys():
                del self.m_LandmarkLocal[index[0]]
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

        return Nmarg, bmarg


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
            pointID = mappoint.m_id
            pointPos = mappoint.m_pos
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
