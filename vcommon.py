import numpy as np
import math
from scipy.spatial.transform import Rotation as R


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


class Feature:
    def __init__(self, pos = np.zeros([2, 1]), du = 0, mapPointId = -1):
        self.m_pos = pos                    # pixels
        self.m_du = du           
        self.m_mapPointId = mapPointId      # 该像素对应的地图点（三维点）Id
        self.m_mappoint = None
        self.m_frame = None


class MapPoint:
    def __init__(self, pos = np.zeros([3, 1]), mapPointId = -1):
        self.m_pos  = pos                       # 三维点世界坐标系的坐标
        self.m_id   = mapPointId                # ID              
        self.m_obs  = []                        # observations (features)


class Map:
    def __init__(self):
        self.m_points = {}                      # observations


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


def rot2att(rotation):
    """Transform rotation to euler angles (Yaw Pitch Roll)

    Args:
        rotation (_type_): _description_

    Returns:
        _type_: _description_
    """
    r = R.from_dcm(rotation)
    return r.as_euler('xyz', degrees=False)

def att2rot(att):
    """Transform euler angle to rotation matrix

    Args:
        att (_type_): _description_
    """
    r = R.from_euler("xyz", att)
    return r.as_dcm()


# def att2Rbe(att, xyz)
