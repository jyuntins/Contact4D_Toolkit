"""Optional high-quality shaded mesh rendering (item 3, "pyrender" backend).

Physically-lit rendering via `pyrender` + `trimesh`, for nicer qualitative
figures than the flat-shaded `contact4d.mesh_render_fast` path. `pyrender`
and `trimesh` are imported lazily inside `render_meshes_pyrender`, so
importing `contact4d` (or using the "fast" backend) never requires them to
be installed -- they're an optional extra (see requirements.txt).

**pyrender only supports pinhole cameras**, so `image` must already be
undistorted -- pass `contact4d.undistort.undistort_image(image, camera)`
(its *default* target intrinsics, i.e. no custom `target_intrinsics`/
`balance`/`fov_scale`), which uses the same `fx/fy/cx/cy` this module's
`_pinhole_params`/`IntrinsicsCamera` do. Passing the raw distorted frame
instead still "works" but the mesh only lines up near the principal point,
increasingly drifting from the background towards the edges of a wide-FOV
frame. This mirrors the internal Contact4D pipeline's
`EgoExoScene.render_smpl`/`render_mano`/`render_keypoints`, which always
render onto `AriaCamera`/`ExoCamera.read_undistorted_image()`.

This function does **not** undistort `image` itself -- do it once yourself
(e.g. in `scripts/visualize_models.py`, before its per-model-kind loop). If
it undistorted internally, calling this once per model kind on the
previous call's output (SMPL, then SMPL-X, then MANO, all composited onto
the same running `image`) would re-undistort an already-undistorted image
each time, compounding into a badly warped result.

**Aria native-frame rotation**: like `contact4d.undistort`, every Aria mode
(`rgb`, `left`, `right`) needs rendering in the camera's native (unrotated)
sensor frame -- `camera.intrinsics`' `cu`/`cv` are defined there, not in the
as-shipped upright frame `image` is in -- then rotated back before
compositing onto `image`. See `contact4d.cameras.aria_fisheye` /
`contact4d.undistort` module docstrings for why this applies to `rgb` too.

Rendering style: a flat accent-blue mesh color, a two-light (no ambient
term) rig, and alpha-channel (not binary depth-mask) compositing for
anti-aliased mesh edges.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np

from .cameras import ARIA_FISHEYE, EXO_FISHEYE, Camera

_DEFAULT_MESH_COLOR = (0.10, 0.35, 1.00)  # accent blue, RGB in [0,1]
_DEFAULT_OPACITY = 0.6

# OpenGL/pyrender cameras look down -Z with Y up; our world_to_camera is
# OpenCV convention (X right, Y down, Z forward) -- this flips Y and Z.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

# Two directional lights (angle_deg, axis), no ambient term.
_LIGHT_ROTATIONS = ((-45.0, (1.0, 0.0, 0.0)), (45.0, (0.0, 1.0, 0.0)))


def _pinhole_params(camera: Camera) -> Tuple[float, float, float, float]:
    if camera.model == EXO_FISHEYE:
        fx, fy, cx, cy = camera.intrinsics[:4]
        return float(fx), float(fy), float(cx), float(cy)
    f, cu, cv_ = camera.intrinsics[:3]
    return float(f), float(f), float(cu), float(cv_)


def render_meshes_pyrender(
    image: np.ndarray,
    vertices_by_key: Dict[str, np.ndarray],
    faces: np.ndarray,
    camera: Camera,
    mesh_color: Tuple[float, float, float] = _DEFAULT_MESH_COLOR,
    alpha: float = _DEFAULT_OPACITY,
    z_near: float = 0.05,
    z_far: float = 20.0,
) -> np.ndarray:
    """Overlay one or more shaded meshes onto `image` using pyrender.

    Arguments mirror `contact4d.mesh_render_fast.render_meshes_fast`.
    Requires the optional `pyrender`/`trimesh` dependencies to be installed.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    height, width = image.shape[:2]
    fx, fy, cx, cy = _pinhole_params(camera)

    # Aria's native (unrotated) sensor frame -- where cu/cv live -- is
    # rotated 90 degrees from `image`'s as-shipped upright frame for every
    # mode (rgb included). Render at the native canvas size, then rotate the
    # result back before compositing onto `image`.
    needs_rotation = camera.model == ARIA_FISHEYE
    render_width, render_height = (height, width) if needs_rotation else (width, height)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=np.zeros(3))
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2, alphaMode="OPAQUE", baseColorFactor=(*mesh_color, 1.0),
    )
    camera_pose = np.linalg.inv(_CV_TO_GL @ camera.world_to_camera)

    for _, vertices in sorted(vertices_by_key.items()):
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False))

    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=z_near, zfar=z_far), pose=camera_pose)
    for angle_deg, axis in _LIGHT_ROTATIONS:
        light_pose = trimesh.transformations.rotation_matrix(np.radians(angle_deg), axis)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=camera_pose @ light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=render_width, viewport_height=render_height)
    try:
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    if needs_rotation:
        rgba = cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)

    bgr = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    mesh_alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    composed = mesh_alpha * bgr + (1 - mesh_alpha) * image.astype(np.float32)
    return composed.astype(np.uint8)
