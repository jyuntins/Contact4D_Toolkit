"""Aria RadTanThinPrism fisheye camera model ("aria_fisheye").

15 intrinsic parameters, in order:
``f, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3``
(one shared focal length; 6 radial, 2 tangential, 4 thin-prism terms).

This is the model used by every Aria egocentric stream in Contact4D's
``metric_extrinsics`` export (``camera_model:
"ARIA_RADTAN_THIN_PRISM_FISHEYE_15"``). It is Meta's Aria camera calibration
model, re-implemented here from scratch in plain numpy -- **no
`projectaria_tools` SDK dependency is required** to project points with it.

Contact4D's ``rgb`` stream is stored square (no rotation ambiguity). The
``left``/``right`` (SLAM) streams are stored rotated 90 degrees from the
camera's native sensor frame; call `rotate_to_upright` after projecting to
match the pixel coordinates of the as-shipped images.
"""

from __future__ import annotations

import numpy as np

NUM_INTRINSICS = 15
_NUM_RADIAL = 6
_START_K = 3
_START_P = _START_K + _NUM_RADIAL  # 9
_START_S = _START_P + 2  # 11


def project_cam_points(points_cam: np.ndarray, intrinsics: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project camera-frame points to raw-device-frame pixel coordinates.

    Args:
        points_cam: ``(N, 3)`` points already in this camera's frame (see
            ``Camera.world_to_cam``).
        intrinsics: ``(15,)`` RadTanThinPrism parameter vector, see module
            docstring for the parameter order.

    Returns:
        ``(N, 2)`` pixel coordinates in the camera's native (unrotated)
        frame. Points behind the camera (``z <= 0``) are returned as
        ``(-1, -1)``. For ``left``/``right`` streams, pass the result through
        `rotate_to_upright` to match the as-shipped image orientation.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (NUM_INTRINSICS,):
        raise ValueError(f"aria_fisheye expects {NUM_INTRINSICS} intrinsics, got {intrinsics.shape}")

    behind = points_cam[:, 2] <= 0
    z_safe = np.where(behind, 1.0, points_cam[:, 2])
    inv_z = 1.0 / z_safe
    ab = points_cam[:, :2] * inv_z.reshape(-1, 1)

    r_sq = (ab**2).sum(axis=1)
    r = np.sqrt(r_sq)
    theta = np.arctan(r)
    theta_sq = theta**2

    radial = np.ones(len(points_cam))
    theta_pow = theta_sq.copy()
    for i in range(_NUM_RADIAL):
        radial += theta_pow * intrinsics[_START_K + i]
        theta_pow = theta_pow * theta_sq

    safe_r = np.where(r > eps, r, 1.0)
    theta_over_r = np.where(r > eps, theta / safe_r, 1.0)
    xr_yr = (radial * theta_over_r).reshape(-1, 1) * ab
    xr_yr_sq_norm = (xr_yr**2).sum(axis=1)

    p = intrinsics[_START_P:_START_P + 2]
    uv_distorted = xr_yr + 2.0 * xr_yr * (xr_yr * p) + xr_yr_sq_norm.reshape(-1, 1) * p

    s = intrinsics[_START_S:_START_S + 4]
    radial_powers_2_4 = np.stack([xr_yr_sq_norm, xr_yr_sq_norm**2], axis=1)
    uv_distorted[:, 0] += (s[0:2] * radial_powers_2_4).sum(axis=1)
    uv_distorted[:, 1] += (s[2:4] * radial_powers_2_4).sum(axis=1)

    points_2d = intrinsics[0] * uv_distorted + intrinsics[1:3]
    points_2d[behind] = -1.0
    return points_2d


def rotate_to_upright(points_2d: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Rotate raw-device-frame pixel coordinates to the as-shipped image frame.

    Only needed for the non-square Aria `left`/`right` streams (`rgb` is
    square and this is a no-op for it up to relabeling). `image_width`/
    `image_height` must be the *as-shipped* (upright) image size, exactly as
    stored in ``metric_extrinsics/<aria>/<mode>.npz``'s ``image_width`` /
    ``image_height``.
    """
    del image_height  # only the (rotated) width is needed, kept for clarity/symmetry
    points_2d = np.asarray(points_2d, dtype=np.float64).copy()
    x = points_2d[:, 0].copy()
    y = points_2d[:, 1].copy()
    rotated_image_height = image_width
    points_2d[:, 0] = rotated_image_height - y
    points_2d[:, 1] = x
    return points_2d
