"""Convert global (world-space) SMPL / SMPL-X / MANO annotations to camera space (item 2).

``processed_data/{smpl,smplx,mano}`` store body/hand model parameters
in the same shared metric world frame as everything else in Contact4D. This
module re-expresses them in one camera's frame instead, given that camera's
rigid ``world_to_camera`` transform (from
``processed_data/camera_params``).

- SMPL / SMPL-X: `global_orient` is rotated and `transl` is re-derived about
  the shape-dependent *canonical pelvis* (the root joint location at zero
  pose), then vertices/joints are regenerated with a body-model forward
  pass and cross-checked against a direct rigid transform of the original
  world-space vertices/joints.
- MANO: root orientation, translation, and root-centered geometry are
  rotated directly (local joint rotations and shape stay unchanged) --
  MANO's stored ``vertices``/``joints`` are the final fit output, so no
  forward pass is needed.

Coordinate convention of the output: camera-native OpenCV axes (x right,
y down, z forward), in meters.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .body_models import BODY_PARAMETER_KEYS, BodyModelCache

DEFAULT_TOLERANCE_M = 2e-5


def _transform_points(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]


def _matrix_to_wxyz_quat(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _validate_rigid(world_to_camera: np.ndarray) -> None:
    world_to_camera = np.asarray(world_to_camera)
    if world_to_camera.shape != (4, 4) or not np.isfinite(world_to_camera).all():
        raise ValueError("world_to_camera must be a finite 4x4 matrix")
    rotation = world_to_camera[:3, :3]
    orthogonality = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    determinant = np.linalg.det(rotation)
    if orthogonality > 1e-6 or abs(determinant - 1.0) > 1e-6:
        raise ValueError(f"world_to_camera is not rigid (orthogonality={orthogonality}, det={determinant})")


def _body_to_camera_space(
    kind: str,
    source: Dict[str, dict],
    world_to_camera: np.ndarray,
    models: BodyModelCache,
    tolerance_m: float,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    _validate_rigid(world_to_camera)
    camera_rotation = world_to_camera[:3, :3]
    camera_translation = world_to_camera[:3, 3]

    exported: Dict[str, dict] = {}
    report: Dict[str, dict] = {}
    for person_name, params in source.items():
        for key in BODY_PARAMETER_KEYS[kind] + ("vertices", "joints"):
            if key not in params:
                raise KeyError(f"{kind}/{person_name} is missing required key {key!r}")

        source_vertices = np.asarray(params["vertices"])
        source_joints = np.asarray(params["joints"])
        if not np.isfinite(source_vertices).all() or not np.isfinite(source_joints).all():
            raise ValueError(f"{kind}/{person_name} has non-finite world-space geometry")

        pelvis = models.canonical_pelvis(kind, params["betas"])
        world_rotation = Rotation.from_rotvec(np.asarray(params["global_orient"]).reshape(3)).as_matrix()
        camera_global_rotation = camera_rotation @ world_rotation
        camera_global_orient = Rotation.from_matrix(camera_global_rotation).as_rotvec()
        camera_transl = (
            camera_rotation @ np.asarray(params["transl"]).reshape(3)
            + camera_translation
            + (camera_rotation - np.eye(3)) @ pelvis
        )

        output = copy.deepcopy(params)
        output["global_orient"] = camera_global_orient.astype(np.float32)
        output["transl"] = camera_transl.astype(np.float32)
        output["canonical_pelvis"] = pelvis.copy()
        output["world_to_camera"] = np.asarray(world_to_camera).copy()

        reconstructed_vertices, reconstructed_joints = models.forward(kind, output)
        direct_vertices = _transform_points(source_vertices, world_to_camera)
        direct_joints = _transform_points(source_joints, world_to_camera)
        vertex_error = np.linalg.norm(reconstructed_vertices - direct_vertices, axis=1)
        joint_error = np.linalg.norm(reconstructed_joints - direct_joints, axis=1)
        max_error = float(max(vertex_error.max(initial=0), joint_error.max(initial=0)))
        if max_error > tolerance_m:
            raise ValueError(
                f"{kind}/{person_name} camera-space reconstruction error {max_error:.3g} m exceeds {tolerance_m:.3g} m"
            )

        output["vertices"] = reconstructed_vertices
        output["joints"] = reconstructed_joints
        validation = {
            "vertex_mean_error_m": float(vertex_error.mean()),
            "vertex_max_error_m": float(vertex_error.max()),
            "joint_mean_error_m": float(joint_error.mean()),
            "joint_max_error_m": float(joint_error.max()),
        }
        output["camera_space_validation"] = validation
        exported[person_name] = output
        report[person_name] = validation
    return exported, report


def smpl_to_camera_space(
    smpl: Dict[str, dict],
    world_to_camera: np.ndarray,
    models: BodyModelCache,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Convert a world-space ``smpl`` annotation dict (as loaded from
    ``processed_data/smpl/<frame>.npy``) to camera space.

    Returns ``(camera_space_smpl, validation_report)``.
    """
    return _body_to_camera_space("smpl", smpl, world_to_camera, models, tolerance_m)


def smplx_to_camera_space(
    smplx_annotation: Dict[str, dict],
    world_to_camera: np.ndarray,
    models: BodyModelCache,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Convert a world-space ``smplx`` annotation dict to camera space.

    Returns ``(camera_space_smplx, validation_report)``.
    """
    return _body_to_camera_space("smplx", smplx_annotation, world_to_camera, models, tolerance_m)


def mano_to_camera_space(
    mano_params: Dict[str, dict],
    world_to_camera: np.ndarray,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Convert a world-space ``mano`` annotation dict (as loaded from
    ``processed_data/mano/<frame>.npy``) to camera space.

    Unlike SMPL/SMPL-X this needs no body-model forward pass: MANO's stored
    ``vertices``/``joints`` are rotated and re-centered directly. Returns
    ``(camera_space_mano, validation_report)``.
    """
    _validate_rigid(world_to_camera)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]

    required = ("betas", "phis", "pose3d", "global_orient", "hand_pose", "rot_mats", "transl", "vertices", "joints")
    exported: Dict[str, dict] = {}
    report: Dict[str, dict] = {}
    for side, params in mano_params.items():
        for key in required:
            if key not in params:
                raise KeyError(f"mano/{side} is missing required key {key!r}")

        source_vertices = np.asarray(params["vertices"])
        source_joints = np.asarray(params["joints"])
        source_transl = np.asarray(params["transl"]).reshape(3)
        root_vertices = np.asarray(params.get("vertices_root_centered", source_vertices - source_transl))
        root_joints = np.asarray(params.get("joints_root_centered", source_joints - source_transl))

        camera_transl = rotation @ source_transl + translation
        camera_root_vertices = root_vertices @ rotation.T
        camera_root_joints = root_joints @ rotation.T
        reconstructed_vertices = camera_root_vertices + camera_transl
        reconstructed_joints = camera_root_joints + camera_transl

        direct_vertices = _transform_points(source_vertices, world_to_camera)
        direct_joints = _transform_points(source_joints, world_to_camera)
        vertex_error = np.linalg.norm(reconstructed_vertices - direct_vertices, axis=1)
        joint_error = np.linalg.norm(reconstructed_joints - direct_joints, axis=1)
        max_error = float(max(vertex_error.max(initial=0), joint_error.max(initial=0)))
        if max_error > tolerance_m:
            raise ValueError(
                f"mano/{side} camera-space reconstruction error {max_error:.3g} m exceeds {tolerance_m:.3g} m"
            )

        pose3d = np.asarray(params["pose3d"]).copy()
        pose3d[:, :3] = pose3d[:, :3] @ rotation.T + translation
        rot_mats = np.asarray(params["rot_mats"]).copy()
        rot_mats[0] = rotation @ rot_mats[0]
        rot_quats = np.stack([_matrix_to_wxyz_quat(matrix) for matrix in rot_mats])

        output = copy.deepcopy(params)
        output["pose3d"] = pose3d
        output["global_orient"] = rot_mats[0]
        output["hand_pose"] = rot_mats[1:]
        output["rot_mats"] = rot_mats
        output["rot_quats"] = rot_quats
        output["transl"] = camera_transl
        output["vertices"] = reconstructed_vertices
        output["joints"] = reconstructed_joints
        output["vertices_root_centered"] = camera_root_vertices
        output["joints_root_centered"] = camera_root_joints
        if "root_joint" in params:
            output["root_joint"] = np.asarray(params["root_joint"]) @ rotation.T
        output["world_to_camera"] = np.asarray(world_to_camera).copy()

        validation = {
            "parameterization": "rotated_root_centered_geometry_plus_camera_translation",
            "vertex_mean_error_m": float(vertex_error.mean()),
            "vertex_max_error_m": float(vertex_error.max()),
            "joint_mean_error_m": float(joint_error.mean()),
            "joint_max_error_m": float(joint_error.max()),
        }
        output["camera_space_validation"] = validation
        exported[side] = output
        report[side] = validation
    return exported, report
