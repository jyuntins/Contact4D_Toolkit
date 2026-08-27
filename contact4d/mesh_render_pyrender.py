"""Optional high-quality shaded mesh rendering (item 3, "pyrender" backend).

Physically-lit rendering via `pyrender` + `trimesh`, for nicer qualitative
figures than the flat-shaded `contact4d.mesh_render_fast` path. `pyrender`
and `trimesh` are imported lazily inside `render_meshes_pyrender`, so
importing `contact4d` (or using the "fast" backend) never requires them to
be installed -- they're an optional extra (see requirements.txt).

**Caveat**: `pyrender` only supports pinhole cameras. This backend
approximates the fisheye camera with a pinhole using the same focal
length/principal point, which is accurate near the image center but
increasingly wrong towards the edges of a wide-FOV fisheye frame. For
geometrically accurate edge-to-edge rendering, undistort the image first
(`contact4d.undistort`) and render onto that instead of the raw frame.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import numpy as np

from .cameras import EXO_FISHEYE, Camera

_DEFAULT_MESH_COLOR = (0.65, 0.74, 0.86)  # RGB in [0,1]

# OpenGL/pyrender cameras look down -Z with Y up; our world_to_camera is
# OpenCV convention (X right, Y down, Z forward) -- this flips Y and Z.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _pinhole_params(camera: Camera) -> Tuple[float, float, float, float]:
    if camera.model == EXO_FISHEYE:
        fx, fy, cx, cy = camera.intrinsics[:4]
        return float(fx), float(fy), float(cx), float(cy)
    f, cu, cv_ = camera.intrinsics[:3]
    return float(f), float(f), float(cu), float(cv_)


def _light_poses(count: int = 4) -> Sequence[np.ndarray]:
    poses = []
    for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
        poses.append(np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]))
    return poses


def render_meshes_pyrender(
    image: np.ndarray,
    vertices_by_key: Dict[str, np.ndarray],
    faces: np.ndarray,
    camera: Camera,
    mesh_color: Tuple[float, float, float] = _DEFAULT_MESH_COLOR,
    alpha: float = 0.9,
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

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.6, baseColorFactor=(*mesh_color, 1.0),
    )
    camera_pose = np.linalg.inv(_CV_TO_GL @ camera.world_to_camera)

    for _, vertices in sorted(vertices_by_key.items()):
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))

    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=z_near, zfar=z_far), pose=camera_pose)
    for light_pose in _light_poses():
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose @ light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    rendered_bgr = color[:, :, :3][:, :, ::-1].astype(np.float32)
    mask = (depth > 0).astype(np.float32)[:, :, None] * alpha
    composed = mask * rendered_bgr + (1 - mask) * image.astype(np.float32)
    return composed.astype(np.uint8)
