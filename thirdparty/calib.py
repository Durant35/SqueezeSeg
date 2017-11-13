"""Calib file from KITTI object detection benchmark, with modifications for KITTI road benchmark"""

import os

import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, 'data')
image_shape = 375, 1242


def get_drive_dir(drive, date='2011_09_26'):
    return os.path.join(data_dir, date, date + '_drive_%04d_sync' % drive)


def get_inds(path, ext='.png'):
    inds = [int(os.path.splitext(name)[0]) for name in os.listdir(path)
            if os.path.splitext(name)[1] == ext]
    inds.sort()
    return inds


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")

    data = {}

    # print(path)

    with open(path, 'r') as f:
        for line in f.readlines():
            # remove the last '\n', and discard the last empty line
            line = line.strip('\n')
            if len(line) == 0:
                continue
            #print (line,'here')
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass

    # print(data)

    return data


def homogeneous_transform(points, transform, keep_last=False):
    """
    Parameters
    ----------
    points : (n_points, M) array-like
        The points to transform. If `points` is shape (n_points, M-1), a unit
        homogeneous coordinate will be added to make it (n_points, M).
    transform : (M, N) array-like
        The right-multiplying transformation to apply.
    """
    points = np.asarray(points)
    transform = np.asarray(transform)
    n_points, D = points.shape
    M, N = transform.shape

    # do transformation in homogeneous coordinates
    if D == M - 1:
        points = np.hstack([points, np.ones((n_points, 1), dtype=points.dtype)])
    elif D != M:
        raise ValueError("Number of dimensions of points (%d) does not match"
                         "input dimensions of transform (%d)." % (D, M))

    new_points = np.dot(points, transform)

    # normalize homogeneous coordinates
    if not keep_last:
        new_points = new_points[:, :-1] / new_points[:, [-1]]
    else:
        new_points = np.hstack(
            [new_points[:, :-1] / new_points[:, [-1]], new_points[:, [-1]]])

    return new_points


def filter_disps(xyd, shape, max_disp=255, return_mask=False):
    x, y, d = xyd.T
    mask = ((x >= 0) & (x <= shape[1] - 1) &
            (y >= 0) & (y <= shape[0] - 1) &
            (d >= 0) & (d <= max_disp))
    xyd = xyd[mask]
    return (xyd, mask) if return_mask else xyd


def filter_depths(xyd, shape):
    x, y, d = xyd.T
    mask = ((x >= 0) & (x <= shape[1] - 1) &
            (y >= 0) & (y <= shape[0] - 1))
    xyd = xyd[mask]
    return xyd


class Calib(object):
    """Convert between coordinate frames.
    This class loads the calibration data from file, and creates the
    corresponding transformations to convert between various coordinate
    frames.
    Each `get_*` function returns a 3D transformation in homogeneous
    coordinates between two frames. All transformations are right-multiplying,
    and can be applied with `homogeneous_transform`.
    """

    def __init__(self, calib_path, color=False):
        self.calib_path = calib_path
        self.data = read_calib_file(self.calib_path)

    def get_imu2velo(self):
        RT_imu2velo = np.eye(4)
        RT_imu2velo[:3, :3] = self.data['R0_rect'].reshape(3, 3)
        RT_imu2velo[:3, 3] = self.data['Tr_imu_to_velo']
        return RT_imu2velo.T

    def get_velo2rect(self):
        RT_velo2cam = np.eye(4)
        RT_velo2cam[:, :3] = self.data['Tr_velo_to_cam'].reshape(3, 4).T

        R_rect00 = np.eye(4)
        R_rect00[:3, :3] = self.data['R0_rect'].reshape(3, 3)

        RT_velo2rect = np.dot(R_rect00, RT_velo2cam)
        return RT_velo2rect.T

    def get_velo2cam(self):
        #RT_velo2cam = np.eye(4)
        RT_velo2cam = self.data['Tr_velo_to_cam'].reshape(3, 4)
        return RT_velo2cam
    def get_cam2road(self):
        RT_velo2cam = self.data['Tr_cam_to_road'].reshape(3, 4)
        return RT_velo2cam
    def get_velo2road(self):
        RT_velo2cam = np.eye(4)
        RT_velo2cam[:, :3] = self.data['Tr_velo_to_cam'].reshape(3, 4).T

        RT_cam2road = np.eye(4)
        RT_cam2road[:, :3] = self.data['Tr_cam_to_road'].reshape(3, 4).T

        RT_velo2road = np.dot(RT_cam2road, RT_velo2cam)
        return RT_velo2road.T

    def get_rect2disp(self):
        cam0, cam1 = (0, 1) if not self.color else (2, 3)
        P_rect0 = self.data['P%d' % cam0].reshape(3, 4)
        P_rect1 = self.data['P%d' % cam1].reshape(3, 4)

        P0, P1, P2 = P_rect0
        Q0, Q1, Q2 = P_rect1

        # create disp transform
        T = np.array([P0, P1, P0 - Q0, P2])
        return T.T

    def get_imu2rect(self):
        return np.dot(self.get_imu2velo(), self.get_velo2rect())

    def get_imu2disp(self):
        return np.dot(self.get_imu2rect(), self.get_rect2disp())

    def get_velo2disp(self):
        return np.dot(self.get_velo2rect(), self.get_rect2disp())

    def get_disp2rect(self):
        return np.linalg.inv(self.get_rect2disp())

    def get_disp2imu(self):
        return np.linalg.inv(self.get_imu2disp())

    def rect2disp(self, points):
        return homogeneous_transform(points, self.get_rect2disp())

    def disp2rect(self, xyd):
        return homogeneous_transform(xyd, self.get_disp2rect())

    def velo2rect(self, points):
        return homogeneous_transform(points, self.get_velo2rect())

    def velo2disp(self, points):
        return homogeneous_transform(points, self.get_velo2disp())

    def imu2rect(self, points):
        return homogeneous_transform(points, self.get_imu2rect())

    def rect2imu(self, points):
        return homogeneous_transform(points, self.get_rect2imu())

    def filter_disps(self, xyd, max_disp=255, return_mask=False):
        return filter_disps(
            xyd, image_shape, max_disp=max_disp, return_mask=return_mask)

    def get_proj(self, cam_idx):
        assert cam_idx == 0 or cam_idx == 1 or cam_idx == 2 or cam_idx == 3, \
            'cam_idx should be 0, 1, 2, or 3 only'
        P = self.data['P%d' % cam_idx].reshape(3, 4)
        return P.T

    def get_velo2depth(self, cam_idx):
        return np.dot(self.get_velo2rect(), self.get_proj(cam_idx))

    def velo2depth(self, points, cam_idx):
        return homogeneous_transform(points, self.get_velo2depth(cam_idx),
                                     keep_last=True)

    def velo2img(self, points, cam_idx):
        return homogeneous_transform(points, self.get_velo2depth(cam_idx))
    def velo2cams(self,points):
        RT_ = self.get_velo2cam()
        points = points.T     
        pnt = np.vstack([points,np.ones(points.shape[1],dtype=points.dtype)])
        xyz_v = np.matmul(RT_,pnt)
        return xyz_v
        #return homogeneous_transform(points, self.get_velo2cam(),keep_last=True)
    def cams2road(self,points):
        RT_ = self.get_cam2road()
        #points = points.T
        print (points.shape)     
        pnt = np.vstack([points,np.ones(points.shape[1],dtype=points.dtype)])
        xyz_v = np.matmul(RT_,pnt)
        return xyz_v
    def velo2road(self,points):
        return self.cams2road(self.velo2cams(points))
    def filter_depths(self, xyd, image_shape):
        return filter_depths(xyd, image_shape)
