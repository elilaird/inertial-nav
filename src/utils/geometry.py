"""
Geometry utilities for SO(3) and SE(3) operations.

This module contains functions for:
- SO(3) exponential map and Jacobians
- Rotation matrix conversions (roll-pitch-yaw <-> rotation matrix)
- Rotation matrix normalization
- Umeyama alignment for point set registration
"""

import numpy as np


# Identity matrices for common dimensions
Id3 = np.eye(3)


def so3_exp(phi):
    """
    SO(3) exponential map: converts rotation vector to rotation matrix.

    Args:
        phi: 3D rotation vector (axis-angle representation)

    Returns:
        3x3 rotation matrix
    """
    angle = np.linalg.norm(phi)

    # Near phi==0, use first order Taylor expansion
    if np.abs(angle) < 1e-8:
        skew_phi = skew_symmetric(phi)
        return np.identity(3) + skew_phi

    axis = phi / angle
    skew_axis = skew_symmetric(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    return c * Id3 + (1 - c) * np.outer(axis, axis) + s * skew_axis


def skew_symmetric(v):
    """
    Convert 3D vector to its skew-symmetric matrix representation.

    Args:
        v: 3D vector [v0, v1, v2]

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def so3_left_jacobian(phi):
    """
    Left Jacobian of SO(3) for use in Lie algebra operations.

    Args:
        phi: 3D rotation vector

    Returns:
        3x3 left Jacobian matrix
    """
    angle = np.linalg.norm(phi)

    # Near |phi|==0, use first order Taylor expansion
    if np.abs(angle) < 1e-8:
        skew_phi = skew_symmetric(phi)
        return Id3 + 0.5 * skew_phi

    axis = phi / angle
    skew_axis = skew_symmetric(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    return (s / angle) * Id3 \
           + (1 - s / angle) * np.outer(axis, axis) \
           + ((1 - c) / angle) * skew_axis


def normalize_rot(Rot):
    """
    Normalize a rotation matrix using SVD to correct numerical drift.

    Ensures the matrix remains in SO(3) by projecting onto the nearest
    proper orthogonal matrix.

    Args:
        Rot: 3x3 rotation matrix (possibly with numerical errors)

    Returns:
        3x3 normalized rotation matrix
    """
    # The SVD is commonly written as a = U S V.H.
    # The v returned by this function is V.H and u = U.
    U, _, V = np.linalg.svd(Rot, full_matrices=False)

    S = np.eye(3)
    S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
    return U.dot(S).dot(V)


def from_rpy(roll, pitch, yaw):
    """
    Convert roll-pitch-yaw angles to rotation matrix.

    Uses ZYX Euler angle convention (yaw -> pitch -> roll).

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        3x3 rotation matrix
    """
    return rotz(yaw).dot(roty(pitch).dot(rotx(roll)))


def rotx(t):
    """
    Elementary rotation matrix around x-axis.

    Args:
        t: Rotation angle (radians)

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """
    Elementary rotation matrix around y-axis.

    Args:
        t: Rotation angle (radians)

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """
    Elementary rotation matrix around z-axis.

    Args:
        t: Rotation angle (radians)

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def to_rpy(Rot):
    """
    Convert rotation matrix to roll-pitch-yaw angles.

    Uses ZYX Euler angle convention.

    Args:
        Rot: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

    if np.isclose(pitch, np.pi / 2.):
        yaw = 0.
        roll = np.arctan2(Rot[0, 1], Rot[1, 1])
    elif np.isclose(pitch, -np.pi / 2.):
        yaw = 0.
        roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
    else:
        sec_pitch = 1. / np.cos(pitch)
        yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                         Rot[0, 0] * sec_pitch)
        roll = np.arctan2(Rot[2, 1] * sec_pitch,
                          Rot[2, 2] * sec_pitch)

    return roll, pitch, yaw


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution for Sim(m) transformation.

    Finds rotation, translation, and optionally scale that best aligns
    two point sets. Based on:

    Umeyama, Shinji: "Least-squares estimation of transformation parameters
    between two point patterns." IEEE PAMI, 1991

    Args:
        x: mxn matrix of points (m = dimension, n = number of points)
        y: mxn matrix of points
        with_scale: If True, also estimate scale (default: False, scale=1.0)

    Returns:
        Tuple of (r, t, c):
            r: mxm rotation matrix
            t: m-dimensional translation vector
            c: scale factor (float)
    """
    # m = dimension, n = nr. of data points
    m, n = x.shape

    # Compute means (eq. 34 and 35)
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # Compute variance (eq. 36)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # Compute covariance matrix (eq. 38)
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (between eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix (eq. 43) - ensures proper rotation (det = +1)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a right-handed coordinate system (Kabsch algorithm)
        s[m - 1, m - 1] = -1

    # Compute rotation (eq. 40)
    r = u.dot(s).dot(v)

    # Compute scale and translation (eq. 42 and 41)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c
