"""OpenCV equidistant fisheye camera model ("exo_fisheye").

8 intrinsic parameters, in order: ``fx, fy, cx, cy, k1, k2, k3, k4``.

This is the model used by every exo (body-worn-external / fixed) camera in
Contact4D's ``camera_params`` export (``camera_model: "OPENCV_FISHEYE"``
in ``processed_data/camera_params/_metadata.json``). It matches OpenCV's
fisheye distortion convention: https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html

The projection formula below is a from-scratch, dependency-free
re-implementation of the equidistant-fisheye forward projection used
internally by the Contact4D capture pipeline (see the dataset's own
processing code, ``lib/datasets/exo_camera.py:vec_image_from_cam``, for the
original), so this package has no dependency on that internal pipeline.
"""

from __future__ import annotations

import numpy as np

NUM_INTRINSICS = 8


def project_cam_points(points_cam: np.ndarray, intrinsics: np.ndarray, max_valid_angle_deg: float = None) -> np.ndarray:
    """Project camera-frame points to pixel coordinates.

    Args:
        points_cam: ``(N, 3)`` points already in this camera's frame (see
            ``Camera.world_to_cam``).
        intrinsics: ``(8,)`` ``[fx, fy, cx, cy, k1, k2, k3, k4]``.
        max_valid_angle_deg: if set, points whose ray angle off the optical
            axis exceeds this are also treated as invalid, in addition to
            ``z <= 0``. See `contact4d.cameras.aria_fisheye.project_cam_points`
            for why this matters (a point in front of the camera at a
            near-90-degree grazing angle can still extrapolate this model's
            radial polynomial far outside its calibrated domain).

    Returns:
        ``(N, 2)`` pixel coordinates. Points behind the camera (``z <= 0``),
        or beyond `max_valid_angle_deg` if given, are returned as ``(-1, -1)``.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (NUM_INTRINSICS,):
        raise ValueError(f"exo_fisheye expects {NUM_INTRINSICS} intrinsics, got {intrinsics.shape}")
    fx, fy, cx, cy, k1, k2, k3, k4 = intrinsics

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    behind = z <= 0
    z_safe = np.where(behind, 1.0, z)

    a = x / z_safe
    b = y / z_safe
    r = np.sqrt(a * a + b * b)
    theta = np.arctan(r)
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

    invalid = behind
    if max_valid_angle_deg is not None:
        invalid = invalid | (theta > np.radians(max_valid_angle_deg))

    safe_r = np.where(r > 1e-9, r, 1.0)
    scale = np.where(r > 1e-9, theta_d / safe_r, 1.0)
    x_prime = scale * a
    y_prime = scale * b

    u = fx * x_prime + cx
    v = fy * y_prime + cy
    points_2d = np.stack([u, v], axis=1)
    points_2d[invalid] = -1.0
    return points_2d
