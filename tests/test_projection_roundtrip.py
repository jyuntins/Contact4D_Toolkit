"""Synthetic-data unit tests -- no real Contact4D sequence is needed.

These check the ported camera-projection and camera-space-conversion math
directly: principal-point sanity, rigidity validation, undistortion
self-consistency at the image center, and MANO camera-space conversion
against an independently-computed direct rigid transform (the same
tolerance-checked pattern used by the internal dataset export tooling).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import camera_space, undistort
from contact4d.cameras import ARIA_FISHEYE, EXO_FISHEYE, Camera
from contact4d.cameras import aria_fisheye, exo_fisheye

EXO_INTRINSICS = np.array([1762.1, 1762.9, 1904.7, 1076.0, 0.0908, -0.0404, 0.0194, -0.0066])
ARIA_RGB_INTRINSICS = np.array(
    [609.94, 717.37, 710.09, 0.4055, -0.4333, -0.0188, 1.3737, -1.8264, 0.6737, 0.000275, -0.000519, 0.000757, -0.000667, 0.00127, -0.000184]
)


def identity_camera(model: str, intrinsics: np.ndarray, mode: str = "rgb") -> Camera:
    return Camera(
        name="test", mode=mode, model=model,
        world_to_camera=np.eye(4), intrinsics=intrinsics,
        image_width=3840, image_height=2160,
    )


def test_exo_fisheye_principal_point_on_axis():
    camera = identity_camera(EXO_FISHEYE, EXO_INTRINSICS)
    point_2d = camera.project(np.array([[0.0, 0.0, 5.0]]))
    assert np.allclose(point_2d[0], EXO_INTRINSICS[2:4], atol=1e-6)


def test_exo_fisheye_behind_camera_is_sentinel():
    point_2d = exo_fisheye.project_cam_points(np.array([[0.1, 0.1, -1.0]]), EXO_INTRINSICS)
    assert np.array_equal(point_2d[0], [-1.0, -1.0])


def test_aria_fisheye_principal_point_on_axis():
    camera = identity_camera(ARIA_FISHEYE, ARIA_RGB_INTRINSICS)
    point_2d = camera.project(np.array([[0.0, 0.0, 5.0]]))
    assert np.allclose(point_2d[0], ARIA_RGB_INTRINSICS[1:3], atol=1e-6)


def test_aria_fisheye_behind_camera_is_sentinel():
    point_2d = aria_fisheye.project_cam_points(np.array([[0.1, 0.1, -1.0]]), ARIA_RGB_INTRINSICS)
    assert np.array_equal(point_2d[0], [-1.0, -1.0])


def test_aria_rotate_to_upright_matches_documented_formula():
    # (image_width, image_height) as upright/as-shipped dims for a left/right stream.
    width, height = 480, 640
    points = np.array([[100.0, 50.0], [10.0, 630.0]])
    rotated = aria_fisheye.rotate_to_upright(points, width, height)
    expected = np.array([[width - 50.0, 100.0], [width - 630.0, 10.0]])
    assert np.allclose(rotated, expected, atol=1e-9)


def test_camera_rejects_wrong_intrinsics_length():
    with pytest.raises(ValueError):
        identity_camera(EXO_FISHEYE, ARIA_RGB_INTRINSICS)


def test_exo_default_target_intrinsics_matches_source_k():
    # Reference: lib/datasets/exo_camera.py:get_undistorted_image passes
    # self.K (not the balance-adjusted self.K_undistorted it computes but
    # never uses) as Knew to cv2.fisheye.undistortImage -- so its actual
    # output uses the same K as the source, unadjusted.
    camera = identity_camera(EXO_FISHEYE, EXO_INTRINSICS)
    _, _, target_intrinsics = undistort.build_undistort_maps(camera)
    fx, fy, cx, cy = EXO_INTRINSICS[:4]
    expected = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    assert np.allclose(target_intrinsics, expected)


def test_aria_default_target_intrinsics_matches_source_k():
    # Reference: lib/datasets/aria_camera.py:init_undistort_map uses its own
    # intrinsics directly as both source and target K (no balance concept).
    camera = identity_camera(ARIA_FISHEYE, ARIA_RGB_INTRINSICS)
    _, _, target_intrinsics = undistort.build_undistort_maps(camera)
    f, cu, cv_ = ARIA_RGB_INTRINSICS[:3]
    expected = np.array([[f, 0, cu], [0, f, cv_], [0, 0, 1]])
    assert np.allclose(target_intrinsics, expected)


def test_undistort_maps_center_ray_hits_source_principal_point():
    # An explicit target K equal to the source's own K, so the only expected
    # error is the <1px rounding to an integer pixel index.
    camera = identity_camera(EXO_FISHEYE, EXO_INTRINSICS)
    fx, fy, cx, cy = EXO_INTRINSICS[:4]
    same_as_source_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    map_x, map_y, target_intrinsics = undistort.build_undistort_maps(camera, target_intrinsics=same_as_source_k)
    center_v, center_u = int(round(target_intrinsics[1, 2])), int(round(target_intrinsics[0, 2]))
    assert abs(map_x[center_v, center_u] - EXO_INTRINSICS[2]) < 1.0
    assert abs(map_y[center_v, center_u] - EXO_INTRINSICS[3]) < 1.0


def _synthetic_mano_params(rng: np.random.Generator) -> dict:
    joints = rng.normal(scale=0.05, size=(21, 3))
    transl = np.array([0.1, -0.2, 1.5])
    rot_mats = np.stack([Rotation.random(random_state=rng).as_matrix() for _ in range(16)])
    return {
        "betas": rng.normal(size=10).astype(np.float32),
        "phis": np.tile([1.0, 0.0], (21, 1)).astype(np.float32),
        "pose3d": np.concatenate([joints + transl, np.ones((21, 1))], axis=1),
        "global_orient": rot_mats[0].astype(np.float32),
        "hand_pose": rot_mats[1:].astype(np.float32),
        "rot_mats": rot_mats.astype(np.float32),
        "transl": transl.astype(np.float32),
        "vertices": (rng.normal(scale=0.05, size=(778, 3)) + transl).astype(np.float32),
        "joints": (joints + transl).astype(np.float32),
    }


def test_mano_to_camera_space_matches_direct_rigid_transform():
    rng = np.random.default_rng(0)
    source = {"left": _synthetic_mano_params(rng), "right": _synthetic_mano_params(rng)}

    rotation = Rotation.random(random_state=rng).as_matrix()
    translation = np.array([0.3, -1.2, 2.0])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation

    output, report = camera_space.mano_to_camera_space(source, world_to_camera)

    for side in ("left", "right"):
        expected_vertices = source[side]["vertices"] @ rotation.T + translation
        expected_joints = source[side]["joints"] @ rotation.T + translation
        assert np.allclose(output[side]["vertices"], expected_vertices, atol=1e-4)
        assert np.allclose(output[side]["joints"], expected_joints, atol=1e-4)
        assert report[side]["vertex_max_error_m"] < camera_space.DEFAULT_TOLERANCE_M


def test_mano_to_camera_space_rejects_non_rigid_transform():
    rng = np.random.default_rng(1)
    source = {"left": _synthetic_mano_params(rng)}
    not_rigid = np.eye(4)
    not_rigid[0, 0] = 2.0
    with pytest.raises(ValueError):
        camera_space.mano_to_camera_space(source, not_rigid)
