import logging
import math
from functools import reduce

import mathutils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import visibility
from matplotlib import cm

_NAME = 'CMRNext'


def init_logger(path, resume=False, save_to_file=False):
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Setup file logging
    if save_to_file:
        mode = 'w'
        if resume:
            mode = 'a'
        file_handler = logging.FileHandler(path, mode=mode)
        file_formatter = logging.Formatter(
            fmt='%(levelname).1s - %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T * R
    else:
        RT = R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=PC.dtype)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError('Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)')
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError('Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)')
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case
            T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contain the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contain the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None
        (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNext to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError('Not a valid rotation matrix')
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr + 1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def euler2mat(z, y, x):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2quat(roll, pitch, yaw):
    q = np.zeros(4)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), 'Not a valid quaternion'
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2 * q[2]**2 - 2 * q[3]**2
    mat[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    mat[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    mat[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    mat[1, 1] = 1 - 2 * q[1]**2 - 2 * q[3]**2
    mat[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    mat[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    mat[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    mat[2, 2] = 1 - 2 * q[1]**2 - 2 * q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), 'Not a valid translation'
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin(rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1, 2, 0).numpy()
    rgb = rgb * std + mean
    lidar = lidar.clone()
    lidar = lidar / lidar.max()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    # lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    lidar = lidar[0][0]
    lidar = (lidar * 255).int().cpu().numpy()
    # pylint: disable-next=no-member
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    return blended_img


def depth_to_3D(uv,
                depth,
                cam_model):
    points_3D = torch.zeros((uv.shape[0], 4), device=uv.device, dtype=torch.float)

    # depth to 3D (Z-forward)
    points_3D[:, 0] = (uv[:, 0] -
                       cam_model.principal_point[0]) * depth / cam_model.focal_length[0]  # Y
    points_3D[:, 1] = (uv[:, 1] -
                       cam_model.principal_point[1]) * depth / cam_model.focal_length[1]  # Z
    points_3D[:, 2] = depth
    points_3D[:, 3] = 1.

    return points_3D


def downsample_depth(depth, downsample_ratio):
    downsampled_depth = torch.zeros(
        (depth.shape[0] // downsample_ratio, depth.shape[1] // downsample_ratio, 1),
        device=depth.device,
        dtype=torch.float)
    downsampled_depth = visibility.downsample_depth(depth, downsampled_depth,
                                                    depth.shape[1] // downsample_ratio,
                                                    depth.shape[0] // downsample_ratio,
                                                    downsample_ratio)
    return downsampled_depth


def show_or_save_plt(img, path, save_images=False):
    if save_images:
        my_dpi = 100
        fig = plt.figure(figsize=(img.shape[1] / my_dpi, img.shape[0] / my_dpi), dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
    # plt.imshow(img)
    if save_images:
        plt.savefig(path, dpi=my_dpi)
        plt.clf()
    else:
        plt.show()


def average_quaternions(quaternions):
    """
    Calculate average quaternion

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    """

    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i, :]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = np.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0 / samples) * mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))
