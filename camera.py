import numpy as np


class Camera:
    def __init__(self, fx, fy, cx, cy, b) -> None:
        self.m_fx                = fx
        self.m_fy                = fy
        self.m_cx                = cx
        self.m_cy                = cy
        self.m_b                 = b
        
    def project(self, xyz):
        """Project point in camera frame to image plane

        Args:
            xyz (ndarray): point in camera frame with shape(3, 1)
        """
        uv = np.zeros((3, 1))
        uv[0, 0] = self.m_fx * xyz[0, 0] / xyz[2, 0] + self.m_cx
        uv[1, 0] = self.m_fy * xyz[1, 0] / xyz[2, 0] + self.m_cy
        uv[2, 0] = self.m_b / xyz[2, 0]

        return uv

    def lift(self, uv):
        """Lift a pixel in image plane to camera frame

        Args:
            uv (ndarray): point in image plane with shape(3, 1)
        """
        K = np.array([[self.m_fx, 0, self.m_cx],
                      [0, self.m_fy, self.m_cy],
                      [0, 0, 1]])
        print(np.linalg.inv(K))
        return np.linalg.inv(K) @ uv
        xyz = np.zeros((3, 1))
        xyz[0, 0] = (uv[0, 0] - self.m_cx) / self.m_fx
        xyz[1, 0] = (uv[1, 0] - self.m_cy) / self.m_fy
        # xyz[2, 0] = self.m_b / uv[2]

        return xyz

