"""Project 3D body/hand keypoints to 2D image coordinates (item 1).

Operates directly on the annotation dict shapes Contact4D ships:
``processed_data/body_pose3d/<frame>.npy`` -> ``{person_id: (17,4)}``
(x, y, z, confidence) for body, and
``processed_data/hand_pose3d/<frame>.npy`` -> ``{"left"/"right": (21,4)}``
for hands. Output mirrors the shipped 2D annotations
(``body_poses2d``, ``hand_poses2d_corrective``): ``{key: (N,3)}`` of
(x, y, confidence) in pixel coordinates, so results can be diffed directly
against them.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .cameras import ARIA_FISHEYE, Camera

# Empirically chosen against a real Contact4D Aria `rgb` calibration (see
# contact4d.cameras.aria_fisheye.project_cam_points's docstring): the
# distortion polynomial stays smooth out to ~65-70 degrees -- past the
# ~70-degree real image corner -- and is already exploding by 80-85. 80
# comfortably covers genuinely visible content while excluding the
# near-90-degree grazing angles (e.g. a wearer's own head/torso joints,
# only centimeters from their own head-mounted camera) that extrapolate the
# polynomial to a wildly out-of-frame pixel coordinate instead of erroring.
DEFAULT_ARIA_MAX_VALID_ANGLE_DEG = 80.0

# Distinguishes "caller didn't pass max_valid_angle_deg" (apply the Aria
# default below) from an explicit `max_valid_angle_deg=None` (caller wants
# clipping off entirely, including for Aria).
_UNSET = object()


def _project_keypoint_dict(keypoints: Dict[str, np.ndarray], camera: Camera, max_valid_angle_deg) -> Dict[str, np.ndarray]:
    if max_valid_angle_deg is _UNSET:
        max_valid_angle_deg = DEFAULT_ARIA_MAX_VALID_ANGLE_DEG if camera.model == ARIA_FISHEYE else None
    projected = {}
    for key, array in keypoints.items():
        array = np.asarray(array, dtype=np.float64)
        points_3d, confidence = array[:, :3], array[:, 3].copy()
        points_2d = camera.project(points_3d, max_valid_angle_deg=max_valid_angle_deg)
        invalid = (points_2d[:, 0] == -1.0) & (points_2d[:, 1] == -1.0)
        confidence[invalid] = 0.0
        projected[key] = np.concatenate([points_2d, confidence.reshape(-1, 1)], axis=1)
    return projected


def project_body_pose3d(
    body_pose3d: Dict[str, np.ndarray], camera: Camera, max_valid_angle_deg=_UNSET
) -> Dict[str, np.ndarray]:
    """Project a ``{person_id: (17,4)}`` body-keypoint dict to 2D.

    `body_pose3d` typically comes from
    ``contact4d.io.load_annotation(.../body_pose3d/<frame>.npy)``.

    By default, for Aria cameras only, joints beyond
    `DEFAULT_ARIA_MAX_VALID_ANGLE_DEG` get confidence forced to 0 (exo is
    left alone) -- see `contact4d.cameras.aria_fisheye.project_cam_points`
    for why a joint technically in front of the camera can still explode to
    a wildly out-of-frame pixel coordinate rather than erroring. Pass an
    explicit angle to override, or `None` to disable clipping entirely
    (including for Aria).
    """
    return _project_keypoint_dict(body_pose3d, camera, max_valid_angle_deg)


def project_hand_pose3d(
    hand_pose3d: Dict[str, np.ndarray], camera: Camera, max_valid_angle_deg=_UNSET
) -> Dict[str, np.ndarray]:
    """Project a ``{"left"/"right": (21,4)}`` hand-keypoint dict to 2D.

    `hand_pose3d` typically comes from
    ``contact4d.io.load_annotation(.../hand_pose3d/<frame>.npy)``.

    Same Aria angle clipping as `project_body_pose3d` -- see there.
    """
    return _project_keypoint_dict(hand_pose3d, camera, max_valid_angle_deg)
