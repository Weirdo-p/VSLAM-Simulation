import numpy as np


class Camera:
    def __init__(self, fx, fy, cx, cy) -> None:
        self.m_fx                = fx
        self.m_fy                = fy
        self.m_cx                = cx
        self.m_cy                = cy
        
    def project(self, xyz):
        """Project point in camera frame to image plane

        Args:
            xyz (ndarray): point in camera frame with shape(3, 1)
        """
        uv = np.zeros((2, 1))
        uv[0, 0] = self.m_fx * xyz[0, 0] / xyz[2, 0] + self.m_cx
        uv[1, 0] = self.m_fy * xyz[1, 0] / xyz[2, 0] + self.m_cy

        return uv

    def lift(self, uv):
        """Lift a pixel in image plane to camera frame

        Args:
            uv (ndarray): point in image plane with shape(2, 1)
        """
        xyz = np.zeros((3, 1))
        xyz[0, 0] = (uv[0, 0] - self.m_cx) / self.m_fx
        xyz[1, 0] = (uv[1, 0] - self.m_cy) / self.m_fy
        xyz[2, 0] = 1

        return xyz

