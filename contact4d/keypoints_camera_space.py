"""Transform 3D keypoints from world space to a camera's frame (item 5).

The simplest of contact4d's tools: a plain rigid transform,
``points_cam = points_world @ R.T + t``, where ``R``/``t`` come from a
``Camera``'s ``world_to_camera``. Kept separate from `projection` (which
also maps camera-frame points on to 2D pixels) since some users only need
the 3D camera-space keypoints -- e.g. for computing distances/contacts in a
single camera's frame without ever touching image coordinates.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .cameras import Camera


def transform_points_to_camera(points_world: np.ndarray, camera: Camera) -> np.ndarray:
    """Rigid-transform ``(N, 3)`` world points into `camera`'s frame."""
    return camera.world_to_cam(points_world)


def transform_keypoints_to_camera(keypoints: Dict[str, np.ndarray], camera: Camera) -> Dict[str, np.ndarray]:
    """Transform a ``{key: (N,4)}`` (x,y,z,confidence) keypoint dict to camera space.

    Works for both the body (``poses3d``/``fit_poses3d``, 17 joints) and
    hand (``pose_corrective``, 21 joints) annotation shapes. The confidence
    column is passed through unchanged.
    """
    transformed = {}
    for key, array in keypoints.items():
        array = np.asarray(array, dtype=np.float64)
        points_cam = camera.world_to_cam(array[:, :3])
        transformed[key] = np.concatenate([points_cam, array[:, 3:4]], axis=1)
    return transformed
