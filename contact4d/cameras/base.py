"""Shared camera abstraction used by every contact4d tool.

A `Camera` is built from one frame of a
``processed_data/metric_extrinsics/<camera>/<mode>.npz`` trajectory (see
``docs/cameras.md``): a rigid world-to-camera transform, the camera's native
intrinsics vector, and the image size. `model` selects which distortion
model interprets `intrinsics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import aria_fisheye, exo_fisheye

EXO_FISHEYE = "exo_fisheye"
ARIA_FISHEYE = "aria_fisheye"

# camera_model strings as stored in metric_extrinsics/_metadata.json
_CAMERA_MODEL_ALIASES = {
    "OPENCV_FISHEYE": EXO_FISHEYE,
    "ARIA_RADTAN_THIN_PRISM_FISHEYE_15": ARIA_FISHEYE,
}

_NUM_INTRINSICS = {
    EXO_FISHEYE: exo_fisheye.NUM_INTRINSICS,
    ARIA_FISHEYE: aria_fisheye.NUM_INTRINSICS,
}


@dataclass
class Camera:
    name: str  # e.g. "cam05" or "aria01"
    mode: str  # e.g. "rgb", "left", "right"
    model: str  # EXO_FISHEYE or ARIA_FISHEYE
    world_to_camera: np.ndarray = field(repr=False)  # (4, 4)
    intrinsics: np.ndarray = field(repr=False)  # (8,) or (15,)
    image_width: int = 0
    image_height: int = 0

    def __post_init__(self) -> None:
        self.world_to_camera = np.asarray(self.world_to_camera, dtype=np.float64).reshape(4, 4)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64).reshape(-1)
        if self.model not in _NUM_INTRINSICS:
            raise ValueError(f"unknown camera model {self.model!r}, expected one of {sorted(_NUM_INTRINSICS)}")
        expected = _NUM_INTRINSICS[self.model]
        if self.intrinsics.shape != (expected,):
            raise ValueError(f"{self.model} camera expects {expected} intrinsics, got {self.intrinsics.shape}")

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation block of world_to_camera."""
        return self.world_to_camera[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """3-vector translation block of world_to_camera."""
        return self.world_to_camera[:3, 3]

    def world_to_cam(self, points_world: np.ndarray) -> np.ndarray:
        """Rigid transform: world-frame ``(N, 3)`` -> camera-frame ``(N, 3)``."""
        points_world = np.asarray(points_world, dtype=np.float64)
        return points_world @ self.rotation.T + self.translation

    def cam_to_image(self, points_cam: np.ndarray) -> np.ndarray:
        """Project camera-frame ``(N, 3)`` points -> pixel ``(N, 2)``."""
        if self.model == EXO_FISHEYE:
            return exo_fisheye.project_cam_points(points_cam, self.intrinsics)
        points_2d = aria_fisheye.project_cam_points(points_cam, self.intrinsics)
        if self.mode in ("left", "right"):
            points_2d = aria_fisheye.rotate_to_upright(points_2d, self.image_width, self.image_height)
        return points_2d

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World-frame ``(N, 3)`` points -> pixel ``(N, 2)``, the common case."""
        return self.cam_to_image(self.world_to_cam(points_world))

    @classmethod
    def from_metric_extrinsics_frame(
        cls,
        name: str,
        mode: str,
        camera_model: str,
        world_to_camera: np.ndarray,
        intrinsics: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> "Camera":
        """Build a Camera from one frame's worth of metric_extrinsics arrays.

        `camera_model` is the raw string from ``metric_extrinsics/<camera>/
        _metadata.json``'s ``camera_metadata[...]["camera_model"]`` (e.g.
        ``"OPENCV_FISHEYE"``); it is mapped to this package's model names.
        """
        model = _CAMERA_MODEL_ALIASES.get(camera_model, camera_model)
        return cls(
            name=name,
            mode=mode,
            model=model,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            image_width=int(image_width),
            image_height=int(image_height),
        )
