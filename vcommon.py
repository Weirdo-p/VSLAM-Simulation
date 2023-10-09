from operator import matmul
from select import select
from unittest import TestResult
import numpy as np
import math
import copy
from camera import *
from scipy.linalg import fractional_matrix_power


PI = math.pi

D2R = PI / 180.0
R2D = 180 / PI
NOISE_POS_STD = 0.001                # noise added on position(meter)
NOISE_ATT_STD = 0.001 * D2R          # noise added on attitudes(rad)

class Frame:
    def __init__(self, id = 0, pos = np.zeros([3, 1]), rotation = np.identity(3)):
        self.m_pos       =  pos                  # position p_{ec}^{e}
        self.m_rota      =  rotation             # rotation R_{e}^{c}
        self.m_posNoise  =  np.zeros([3, 1])     # position with noise
        self.m_rotaNoise =  np.identity(3)       # rotation with noise
        self.m_id        =  id
        self.m_features  =  []                   # features
        self.m_time      = 0
        self.m_cov       = np.identity(6)        # posteriori 
        self.m_nees      = 0                     # quantity used for consistency

    def __deepcopy__(self, memo):
        if self in memo:
            return memo.get(self)

        dup = Frame(self.m_id, copy.deepcopy(self.m_pos, memo), copy.deepcopy(self.m_rota, memo))
        memo[self] = dup
        dup.m_features = copy.deepcopy(self.m_features, memo)
        dup.m_time = self.m_time
        return dup
    
    def check(self):
        valid = 0
        for feat in self.m_features:
            if feat.m_buse and feat.m_mappoint.m_buse == 1:
                valid += 1
        
        return valid
    
    def reset(self):
        self.m_pos = np.zeros((3, 1))
        self.m_rota = np.identity(3)

        for feat in self.m_features:
            feat.m_buse = True
            feat.m_mappoint.m_buse = True

class Feature:
    def __init__(self, pos = np.zeros([3, 1]), du = 0, mapPointId = -1):
        self.m_pos = pos                    # pixels
        self.m_PosInCamera = np.zeros([3, 1])
        self.m_du = du           
        self.m_mapPointId = mapPointId      # 该像素对应的地图点（三维点）Id
        self.m_mappoint = None
        self.m_frame = None
        self.m_btriangulate = False
        self.m_buse = True

    def __deepcopy__(self, memo):
        if self in memo:
            return memo.get(self)
        
        dup = Feature(copy.deepcopy(self.m_pos, memo), self.m_du, self.m_mapPointId)
        memo[self] = dup

        dup.m_PosInCamera = self.m_PosInCamera.copy()
        if self.m_mappoint is not None:
            dup.m_mappoint = copy.deepcopy(self.m_mappoint, memo)
        if self.m_frame is not None:
            dup.m_frame = copy.deepcopy(self.m_frame, memo)
        dup.m_btriangulate =  self.m_btriangulate
        dup.m_buse = self.m_buse

        return dup
        

class MapPoint:
    def __init__(self, pos = np.zeros([3, 1]), mapPointId = -1):
        self.m_pos  = pos                       # 三维点世界坐标系的坐标
        self.m_id   = mapPointId                # ID              
        self.m_obs  = []                        # observations (features)
        self.m_buse = 1                         # 0 and -1: not use, >= 1: use
        self.m_bconstrain = False

    def __deepcopy__(self, memo):
        if self in memo:
            return memo.get(self)        

        dup = MapPoint(copy.deepcopy(self.m_pos, memo), self.m_id)
        dup.m_buse = self.m_buse
        memo[self] = dup

        dup.m_obs = copy.deepcopy(self.m_obs, memo)
        return dup

    def check(self):
        valid = 0
        for obs in self.m_obs:
            if obs.m_buse == True:
                valid += 1
        
        if valid < 2:
            self.m_buse = 0
        else:
            self.m_buse = 1

class Map:
    def __init__(self):
        self.m_points = {}                      # observations
        self.m_frames = []                      # frames
        self.m_bFirstFrame = True
        self.m_camera = Camera(0, 0, 0, 0, 0)

    def __parseLine(self, line=str()):
        items = line.split()

        id = int(items[0])
        x, y, z = float(items[1]), float(items[2]), float(items[3])

        mappoint = MapPoint(np.array([[x], [y], [z]]), id)
        return mappoint

    def clear(self):
        for id, point in self.m_points.items():
            point.m_buse = 1
        
        for frame in self.m_frames:
            for feat in frame.m_features:
                feat.m_buse = True
        # self.m_points.clear()
        # self.m_frames.copy()
        self.m_bFirstFrame = True


    def readMapFile(self, path):
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    mappoint = self.__parseLine(line)
                    if mappoint.m_id == -1:
                        continue
                    self.m_points[mappoint.m_id] = mappoint
                    # print(line)
                else:
                    break
    
    def addNewFrame(self, frame, gmap=None):
        # 1. add frames
        self.m_frames.append(frame)

        # 2. add landmarks
        count = 0
        newpoint = 0
        for feat in frame.m_features:
            pointID = feat.m_mapPointId
            if pointID not in self.m_points.keys():
                mappoint = MapPoint()
                mappoint.m_id = pointID
                self.m_points[pointID] = mappoint
                newpoint += 1

            self.m_points[pointID].m_obs.append(feat)
            feat.m_mappoint = self.m_points[pointID]
            if gmap is not None and pointID in gmap.m_points.keys():
                self.m_points[pointID].m_pos = gmap.m_points[pointID].m_pos.copy()
                continue
            # triangulate points in camera frame
            if feat.m_btriangulate == False:
                # points in world frame equals to points in 
                # camera frame if it is the first frame
                if (self.TriangulateByStereo(feat) == False):
                    feat.m_mappoint.m_buse = -1
                else:
                    feat.m_btriangulate = True
                    # feat.m_mappoint.m_buse = True
                    count += 1
                    if self.m_bFirstFrame:
                        self.m_points[pointID].m_pos = feat.m_PosInCamera
        # print("frame ID: ", frame.m_time, " ", count, " landmarks triangulated,", newpoint, "new found")
        if len(self.m_frames) >= 2:
            self.check()

        self.m_bFirstFrame = False
    
    def check(self):
        useful = 0
        for id, point in self.m_points.items():
            if point.m_buse == -1:
                continue
            if len(point.m_obs) < 2: # or point.m_bconstrain == False:
                self.m_points[id].m_buse = 0
            else:
                self.m_points[id].m_buse = 1
                useful += 1
            # point.check()

    def TriangulateByStereo(self, feature):
        """Solve points in camera frame(left)

        Args:
            feature (ndarray): vcommon.Feature object
        """
        id = feature.m_id
        
        # baseline: from left to right camera
        baseline = self.m_camera.getBaseline()
        rPc = np.array([[baseline], [0], [0]])
        rRc = np.identity(3)

        lPc = np.zeros((3, 1))
        lRc = np.identity(3)

        luv = np.array((feature.m_pos))
        ruv = np.array((feature.m_pos))
        # print(ruv)
        ruv[0, 0] = ruv[0, 0] - luv[2, 0]
        # print(ruv)

        lxyz = self.m_camera.lift(luv)
        lxyz = lxyz / lxyz[2, 0]

        rxyz = self.m_camera.lift(ruv)
        rxyz = rxyz / rxyz[2, 0]

        Pj = np.zeros((3, 1))

        for i in range(2):
            J, l = np.zeros((4, 3)), np.zeros((4, 1))
            J[0, :] = -lRc[1, :] + lxyz[1, 0] * lRc[2, :]
            J[1, :] =  lRc[0, :] - lxyz[0, 0] * lRc[2, :]
            J[2, :] = -rRc[1, :] + rxyz[1, 0] * rRc[2, :]
            J[3, :] =  rRc[0, :] - rxyz[0, 0] * rRc[2, :]

            l[:2, :] = J[:2, :] @ (Pj - lPc)
            l[2:, :] = J[2:, :] @ (Pj - rPc)

            # print(J)

            dx = np.linalg.inv(J.transpose() @ J) @ J.transpose() @ l
            Pj = Pj - dx

        if (Pj[2, 0] <= 0):
            return False
        # if (Pj[2, 0] >=500):
        #     return False
        feature.m_PosInCamera = Pj
        
        return True

    def triangulate(self):
        count = 0
        for id, point in self.m_points.items():
            if len(point.m_obs) > 2 and point.m_buse != 1:
                if self.TriangulateOnePoint(point) == False:
                    point.m_buse = -1
                    continue
                point.m_buse = 1
                count += 1
        
        return count

    def TriangulateOnePoint(self, point):
        feats = point.m_obs
        baseline = self.m_camera.getBaseline()
        rPc = np.array([[baseline], [0], [0]])
        Pj = np.zeros((3, 1))
        MapPoint().m_id

        for i in range(2):
            N, b = np.zeros((3, 3)), np.zeros((3, 1))
            for feat in feats:
                lPcam, lRcam = feat.m_frame.m_pos, feat.m_frame.m_rota
                rPcam = lPcam + np.linalg.inv(lRcam) @ rPc
                rRcam = lRcam.copy()

                luv = np.array((feat.m_pos))
                ruv = np.array((feat.m_pos))
                ruv[0, 0] = ruv[0, 0] - luv[2, 0]

                lxyz = self.m_camera.lift(luv)
                lxyz = lxyz / lxyz[2, 0]
                rxyz = self.m_camera.lift(ruv)
                rxyz = rxyz / rxyz[2, 0]

                J, l = np.zeros((4, 3)), np.zeros((4, 1))
                J[0, :] = -lRcam[1, :] + lxyz[1, 0] * lRcam[2, :]
                J[1, :] =  lRcam[0, :] - lxyz[0, 0] * lRcam[2, :]
                J[2, :] = -rRcam[1, :] + rxyz[1, 0] * rRcam[2, :]
                J[3, :] =  rRcam[0, :] - rxyz[0, 0] * rRcam[2, :]

                l[:2, :] = J[:2, :] @ (Pj - lPcam)
                l[2:, :] = J[2:, :] @ (Pj - rPcam)

                N += J.transpose() @ J
                b += J.transpose() @ l
            dx = np.linalg.inv(N) @ b
            Pj = Pj - dx
        
        for feat in feats:
            feat = feats[0]
            lPcam, lRcam = feat.m_frame.m_pos, feat.m_frame.m_rota
            pcam = lRcam @ (Pj - lPcam)
            if (pcam[2, 0] <= 0):
                return False
            if (pcam[2, 0] >= 300):
                return False
        
        point.m_pos = Pj
        return True



def rot2att(rotation):
    """Transform rotation to euler angles (Yaw Pitch Roll)

    Args:
        rotation (_type_): _description_

    Returns:
        _type_: _description_
    """
    pitch = np.arcsin(rotation[2, 1])
    roll = np.arctan2(-rotation[2, 0], rotation[2, 2])
    yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])

    # if yaw * R2D > 150:
    #     yaw -= PI
    # elif yaw * R2D < -150:
    #     yaw += PI
    
    return np.array([yaw, pitch, roll])

def att2rot(att):
    """Transform euler angle to rotation matrix

    Args:
        att (ndarray): [yaw pitch roll] in radians
    """
    # R = np.zeros((3, 3))

    # i, j, k -> p r y
    sin, cos = np.sin(att), np.cos(att)
    sk, si, sj = sin[0], sin[1], sin[2]
    ck, ci, cj = cos[0], cos[1], cos[2]

    R = np.array([[ cj*ck-si*sj*sk, -ci*sk,  sj*ck+si*cj*sk],
                  [ cj*sk+si*sj*ck,  ci*ck,  sj*sk-si*cj*ck],
                  [-ci*sj,           si,     ci*cj         ]])
    
    return R



def SkewSymmetricMatrix(vector):
    """transfer to Skew-Symmetric Matrix

    Args:
        vector (ndarray): vector with shape(3, 1)

    Returns:
        ndarray: Skew-Symmetric Matrix with shape (3, 3)
    """
    matrix = np.zeros((3, 3))

    matrix[0, 1] = -vector[2]
    matrix[0, 2] = vector[1]

    matrix[1, 0] = vector[2]
    matrix[1, 2] = -vector[0]

    matrix[2, 0] = -vector[1]
    matrix[2, 1] = vector[0]

    return matrix

def Jacobian_phai(Rec, tec, XYZ, xyz, fx, fy, b):
    """Compute jacobian matrix to attitude

    Args:
        Rec (ndarray): rotation from e-frame to c-frame
        tec (ndarray): position of frame in e-frame
        XYZ (ndarray): position of mappoint in e-frame
        xyz (ndarray): position of mappoint in c-frame
        fx (float): focal length
        fy (float): focal length
        b (float): fx * baseline
    
    Returns:
        Jphi  (ndarray): Jacobian matrix with shape(3, 3)
    """

    Jpoint = Jacobian_ppoint(fx, fy, b, xyz)
    Jphi = Rec @ SkewSymmetricMatrix(tec - XYZ)

    return matmul(Jpoint, Jphi)

def Jacobian_rcam(Rec, xyz, fx, fy, b):
    Jpoint = Jacobian_ppoint(fx, fy, b, xyz)

    return np.matmul(Jpoint, -Rec)

def Jacobian_Point(Rec, xyz, fx, fy, b):
    Jpoint = Jacobian_ppoint(fx, fy, b, xyz)

    return np.matmul(Jpoint, Rec)

def Jacobian_ppoint(fx, fy, b, xyz):
    Jpoint = np.zeros((3, 3))
    
    x, y, z = xyz[0], xyz[1], xyz[2]
    # u
    Jpoint[0, 0] = fx / z
    Jpoint[0, 2] = -fx * x / z / z

    # v
    Jpoint[1, 1] = fy / z
    Jpoint[1, 2] = -fy * y / z / z

    # du
    Jpoint[2, 2] = - b / z / z

    return Jpoint

def robustKernelHuber(residual, thres):
    if np.any(residual < 1E-4):
        return np.array([1, 1, 1])
    
    P = []
    for i in range(residual.shape[0]):
        w =  Huber(residual[i], thres) / np.linalg.norm(residual[i], 2)
        P.append(w)
    return np.array(P)

def Huber(residual, thres):
    if np.all(np.abs(residual) < thres):
        return np.abs(residual)
    else:
        return np.sqrt(2 * thres * np.abs(residual) - thres ** 2)

def quat2rota(q0, q1, q2, q3):
    q0_square, q1_square, q2_square, q3_square = q0 ** 2, q1 ** 2, q2 ** 2, q3 ** 2
    R = np.array([[q0_square + q1_square - q2_square - q3_square, 2 * (q1 * q2 - q0 * q3)                      , 2 * (q1 * q3 + q0 * q2)],
                  [2 * (q1 * q2 + q0 * q3)                      , q0_square - q1_square + q2_square - q3_square, 2 * (q2 * q3 - q0 * q1)],
                  [2 * (q1 * q3 - q0 * q2)                      , 2 * (q2 * q3 + q0 * q1)                      , q0_square - q1_square - q2_square + q3_square]]) # .transpose()
    
    return R

def UnitRotation(R):
    return R @ fractional_matrix_power((R.transpose() @ R), -0.5)