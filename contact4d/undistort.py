"""Undistort exo (fisheye) or Aria images into a pinhole projection (item 4).

Works uniformly for both camera models supported by this package via one
generic technique: for every pixel of a *virtual pinhole* target image,
compute the corresponding camera-frame ray and project it through the
*source* camera's real (distorted) forward model to find where to sample
from in the original image, then `cv2.remap`. This avoids relying on
`cv2.fisheye`'s own undistortion (which only implements the exo/OpenCV-
fisheye model) for Aria too -- both camera models are just a
`project_cam_points` function to this one routine.

**Aria native-frame subtlety**: every Aria stream (`rgb`, `left`, `right`)
is shipped pre-rotated 90 degrees from the camera's native sensor
orientation (see `docs/cameras.md`), but the camera's intrinsics (and its
distortion formula) are defined in that *native*, unrotated frame. So the
map has to be built and sampled entirely in the native frame -- the
as-shipped image is rotated into native orientation first, undistorted
there, then rotated back. This exactly mirrors the internal Contact4D
pipeline's `AriaCamera.get_undistorted_image_aria` (rotate CCW -> remap in
native frame -> rotate CW), which applies unconditionally to every Aria
mode, `rgb` included. Building the map directly against upright-frame pixel
indices while using the camera's native-frame principal point (an earlier,
buggy version of this module did that, and additionally skipped the
rotation entirely for `rgb` on the mistaken assumption that being square
made it a no-op) silently misaligns Aria outputs -- only exo streams are
unaffected.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .cameras import ARIA_FISHEYE, Camera, EXO_FISHEYE
from .cameras import aria_fisheye, exo_fisheye


def _needs_native_frame_rotation(camera: Camera) -> bool:
    return camera.model == ARIA_FISHEYE


def _native_canvas_size(camera: Camera) -> Tuple[int, int]:
    """(width, height) of the camera's own distortion-model frame.

    Equal to `(camera.image_width, camera.image_height)` for every exo
    camera; for every Aria mode (`rgb`, `left`, `right`) the as-shipped
    (upright) frame is rotated 90 degrees relative to the native frame the
    intrinsics/distortion model are defined in.
    """
    if _needs_native_frame_rotation(camera):
        return camera.image_height, camera.image_width
    return camera.image_width, camera.image_height


def _default_target_intrinsics(
    camera: Camera, balance: Optional[float], fov_scale: float, native_size: Tuple[int, int]
) -> np.ndarray:
    """A target pinhole K matching what the internal Contact4D undistortion
    tooling's `get_undistorted_image` *actually* produces for each model
    (verified against its real output, not just its computation): reuse the
    source's own focal length and principal point unchanged, for both
    models.

    This looks like it should differ for exo cameras --
    `lib/datasets/exo_camera.py:init_undistort_map` computes a
    balance-adjusted `self.K_undistorted` via
    `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(..., balance=1.0)`
    -- but its own `get_undistorted_image` passes `self.K` (not
    `self.K_undistorted`) as the target `Knew` to `cv2.fisheye.undistortImage`,
    so that balance-adjusted K is computed and never actually used. An
    earlier version of this function used the balance-adjusted K to match
    what looked like the intended behavior, which silently produced a
    different (over-cropped) result than the reference tool's real output.
    `lib/datasets/aria_camera.py:init_undistort_map` has no such
    discrepancy -- it uses its own intrinsics directly as both source and
    target K by construction.

    `balance`/`fov_scale` (both models) additionally zoom the target out
    (`balance` > 0 or `fov_scale` > 1) or in (`fov_scale` < 1) from that
    same-as-source default; pass `target_intrinsics` directly (e.g. computed
    via `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify` yourself)
    if you want a different, non-default crop for exo specifically.
    """
    del native_size  # only used by target_intrinsics recipes callers build themselves
    if camera.model == EXO_FISHEYE:
        fx, fy, cx, cy = camera.intrinsics[:4]
    else:
        fx = fy = camera.intrinsics[0]
        cx, cy = camera.intrinsics[1], camera.intrinsics[2]
    scale = fov_scale * (1.0 + (0.0 if balance is None else balance))
    return np.array([[fx / scale, 0.0, cx], [0.0, fy / scale, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_undistort_maps(
    camera: Camera,
    target_intrinsics: Optional[np.ndarray] = None,
    output_size: Optional[Tuple[int, int]] = None,
    balance: Optional[float] = None,
    fov_scale: float = 1.0,
    max_valid_angle_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `cv2.remap`-compatible maps that undistort images from `camera`.

    The maps operate in `camera`'s *native* distortion-model frame (see
    `_native_canvas_size`) -- for exo cameras this is the same as the
    as-shipped image; for every Aria mode (`rgb`, `left`, `right`), feed
    `cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)` in and rotate the
    remapped result back with `cv2.ROTATE_90_CLOCKWISE` (or just call
    `undistort_image`, which does this for you).

    Args:
        camera: the source (distorted) camera.
        target_intrinsics: pinhole ``3x3`` K for the output image, in the
            native frame. If omitted, a default derived from `camera`'s own
            focal length is used (see `balance`/`fov_scale` and
            `_default_target_intrinsics`).
        output_size: native-frame ``(width, height)`` of the output image;
            defaults to `camera`'s own native canvas size.
        balance: only used for the default `target_intrinsics`; per-model
            meaning documented in `_default_target_intrinsics`. `None` (the
            default) picks each model's own reference-matching value.
        fov_scale: only used for the default `target_intrinsics`.
        max_valid_angle_deg: if set, target rays whose angle off the optical
            axis exceeds this are marked invalid (mapped off-image, so
            `cv2.remap`'s border fill is used) rather than sampled -- both
            fisheye models here are finite-degree polynomials in the ray
            angle that are only calibrated to be monotonic within the
            camera's real field of view, and a target FOV wide enough to
            reach past that range can fold distant, unrelated source pixels
            back into view ("ghosting") instead of legitimately having no
            coverage there. Unset (`None`) by default to match the reference
            tooling's own unclipped output on well-calibrated cameras; if a
            specific camera's intrinsics are themselves miscalibrated (e.g.
            a resolution/principal-point mismatch), passing e.g. `85.0` here
            trades a wider black border for suppressing that artifact.

    Returns:
        ``(map_x, map_y, target_intrinsics)``; ``map_x``/``map_y`` are
        ``float32`` arrays of shape ``(height, width)`` suitable for
        ``cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)``.
    """
    width, height = output_size or _native_canvas_size(camera)
    if target_intrinsics is None:
        target_intrinsics = _default_target_intrinsics(camera, balance, fov_scale, (width, height))
    target_intrinsics = np.asarray(target_intrinsics, dtype=np.float64)

    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    rays = pixels @ np.linalg.inv(target_intrinsics).T  # (N,3) camera-frame rays at z=1

    if camera.model == EXO_FISHEYE:
        source_pixels = exo_fisheye.project_cam_points(rays, camera.intrinsics)
    else:
        # Stays in the native frame throughout -- no rotate_to_upright here;
        # that conversion is handled by rotating the image, not the map.
        source_pixels = aria_fisheye.project_cam_points(rays, camera.intrinsics)

    if max_valid_angle_deg is not None:
        ray_angle = np.arctan(np.sqrt((rays[:, :2] ** 2).sum(axis=1)))  # rays have z=1
        source_pixels = source_pixels.copy()
        source_pixels[ray_angle > np.deg2rad(max_valid_angle_deg)] = -1.0

    map_x = source_pixels[:, 0].reshape(height, width).astype(np.float32)
    map_y = source_pixels[:, 1].reshape(height, width).astype(np.float32)
    return map_x, map_y, target_intrinsics


def undistort_image(
    image: np.ndarray,
    camera: Camera,
    target_intrinsics: Optional[np.ndarray] = None,
    output_size: Optional[Tuple[int, int]] = None,
    balance: Optional[float] = None,
    fov_scale: float = 1.0,
    max_valid_angle_deg: Optional[float] = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    """Undistort one as-shipped image captured by `camera` into a pinhole projection."""
    rotate = _needs_native_frame_rotation(camera)
    source = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotate else image
    map_x, map_y, _ = build_undistort_maps(camera, target_intrinsics, output_size, balance, fov_scale, max_valid_angle_deg)
    result = cv2.remap(source, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    return cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE) if rotate else result
