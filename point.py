import numpy as np


class CartesianPoint:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.length = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        self.elev = np.rad2deg(np.arccos(self.z / self.get_length()))
        self.azim = np.rad2deg(np.arctan2(self.y, self.x))

    def get_length(self):
        return self.length

    def get_elev(self):
        return self.elev

    def get_azim(self):
        return self.azim

    def get_spherical(self):
        return np.array([self.length, self.elev, self.azim])


class SphericalPoint:
    def __init__(self, point):
        self.dist = point[0]
        self.elev = point[1] * np.pi/180  # 0~pi
        self.azim = point[2] * np.pi/180  # 0~2pi
        self.x = self.dist * np.sin(self.elev) * np.cos(self.azim)
        self.y = self.dist * np.sin(self.elev) * np.sin(self.azim)
        self.z = self.dist * np.cos(self.elev)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_cartesian(self):
        return np.array([self.x, self.y, self.z])


if __name__ == '__main__':
    # a = [3, 0.5, 1]
    # print(a)
    # a1 = CartesianPoint(a).get_spherical()
    # print(a1)
    # b1 = SphericalPoint(a1).get_cartesian()
    # print(b1)

    a = [1.74, 0.5, 1]
    print(a)
    a1 = SphericalPoint(a).get_cartesian()
    print(a1)
    b1 = CartesianPoint(a1).get_spherical()
    print(b1)

    # print(1.8254, 0, 0)
    # dist, elev, azim = 1.8254, 90 - 0.0, 0.0
    # xyz = SphericalPoint([dist, elev, azim]).get_cartesian()
    # print(xyz)