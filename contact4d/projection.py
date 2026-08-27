"""Project 3D body/hand keypoints to 2D image coordinates (item 1).

Operates directly on the annotation dict shapes Contact4D ships:
``processed_data/{poses3d,fit_poses3d}/<frame>.npy`` -> ``{person_id: (17,4)}``
(x, y, z, confidence) for body, and
``processed_data/pose_corrective/<frame>.npy`` -> ``{"left"/"right": (21,4)}``
for hands. Output mirrors the shipped 2D annotations
(``body_poses2d``, ``hand_poses2d_corrective``): ``{key: (N,3)}`` of
(x, y, confidence) in pixel coordinates, so results can be diffed directly
against them.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .cameras import Camera


def _project_keypoint_dict(keypoints: Dict[str, np.ndarray], camera: Camera) -> Dict[str, np.ndarray]:
    projected = {}
    for key, array in keypoints.items():
        array = np.asarray(array, dtype=np.float64)
        points_3d, confidence = array[:, :3], array[:, 3]
        points_2d = camera.project(points_3d)
        projected[key] = np.concatenate([points_2d, confidence.reshape(-1, 1)], axis=1)
    return projected


def project_body_pose3d(body_pose3d: Dict[str, np.ndarray], camera: Camera) -> Dict[str, np.ndarray]:
    """Project a ``{person_id: (17,4)}`` body-keypoint dict to 2D.

    `body_pose3d` typically comes from
    ``contact4d.io.load_annotation(.../poses3d/<frame>.npy)`` (or
    ``fit_poses3d``).
    """
    return _project_keypoint_dict(body_pose3d, camera)


def project_hand_pose3d(hand_pose3d: Dict[str, np.ndarray], camera: Camera) -> Dict[str, np.ndarray]:
    """Project a ``{"left"/"right": (21,4)}`` hand-keypoint dict to 2D.

    `hand_pose3d` typically comes from
    ``contact4d.io.load_annotation(.../pose_corrective/<frame>.npy)``.
    """
    return _project_keypoint_dict(hand_pose3d, camera)
