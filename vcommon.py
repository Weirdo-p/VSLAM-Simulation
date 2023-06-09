from operator import matmul
from select import select
from unittest import TestResult
import numpy as np
import math
import copy
from camera import *


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

    def __deepcopy__(self, memo):
        print("frame copy")
        dup = Frame(copy.deepcopy(self.m_id, memo))
        memo[self] = dup
        dup.m_features = copy.deepcopy(self.m_features)
        dup.m_pos = copy.deepcopy(self.m_pos)
        dup.m_rota = copy.deepcopy(self.m_rota)
        dup.m_time = copy.deepcopy(self.m_time)
        return dup



class Feature:
    def __init__(self, pos = np.zeros([3, 1]), du = 0, mapPointId = -1):
        self.m_pos = pos                    # pixels
        self.m_PosInCamera = np.zeros([3, 1])
        self.m_du = du           
        self.m_mapPointId = mapPointId      # 该像素对应的地图点（三维点）Id
        self.m_mappoint = None
        self.m_frame = None
        self.m_btriangulate = False

    def __deepcopy__(self, memo):
        print("feature copy")
        dup = Feature(copy.deepcopy(self.m_mapPointId, memo))
        memo[self] = dup

        dup.m_du = copy.deepcopy(self.m_du)
        dup.m_pos = copy.deepcopy(self.m_pos)
        dup.m_frame = copy.deepcopy(self.m_frame)
        dup.m_mappoint = copy.deepcopy(self.m_mappoint)
        return dup
        

class MapPoint:
    def __init__(self, pos = np.zeros([3, 1]), mapPointId = -1):
        self.m_pos  = pos                       # 三维点世界坐标系的坐标
        self.m_id   = mapPointId                # ID              
        self.m_obs  = []                        # observations (features)
        self.m_bused = False

    def __deepcopy__(self, memo):
        print("mappoint copy")
        dup = MapPoint(copy.deepcopy(self.m_id, memo))
        dup.m_pos = copy.deepcopy(self.m_pos)
        dup.m_obs = copy.deepcopy(self.m_obs)

        return dup


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
    
    def addNewFrame(self, frame):
        # 1. add frames
        self.m_frames.append(frame)

        # 2. add landmarks
        for feat in frame.m_features:
            pointID = feat.m_mapPointId
            if pointID not in self.m_points.keys():
                mappoint = MapPoint()
                mappoint.m_id = pointID
                self.m_points[pointID] = mappoint

            # triangulate points in camera frame
            if feat.m_btriangulate == False:
                # points in world frame equals to points in 
                # camera frame if it is the first frame
                if (self.TriangulateByStereo(feat) and self.m_bFirstFrame):
                    self.m_points[pointID].m_pos = feat.m_PosInCamera

            self.m_points[pointID].m_obs.append(feat)
            feat.m_mappoint = self.m_points[pointID]
        self.m_bFirstFrame = False
    
    def check(self):
        useful = 0
        for id, point in self.m_points.items():
            if len(point.m_obs) >= 2:
                point.m_bused = True
                useful += 1
        return useful

    def TriangulateByStereo(self, feature):
        """Solve points in camera frame(left)

        Args:
            feature (ndarray): vcommon.Feature object
        """
        xyz = self.m_camera.lift(feature.m_pos)

        if (xyz[2, 0] <= 0):
            return False
        feature.m_PosInCamera = xyz
        
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
        att (_type_): _description_
    """
    r = R.from_euler("zxy", att)
    return r.as_dcm()

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