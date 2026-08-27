"""Fast, dependency-free mesh overlay rendering (item 3, "fast" backend).

Flat-shaded triangle rasterization via OpenCV's `fillPoly`, supersampled
for smoother edges -- no OpenGL/EGL setup required, works anywhere. Ported
from the approach used internally to QA MANO fits on this dataset. This
backend does **not** depth-sort/hidden-surface-remove faces within a mesh or
across overlapping meshes; for a physically-shaded, occlusion-correct
render, use `contact4d.mesh_render_pyrender` instead.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .cameras import Camera

_DEFAULT_COLORS = ((128, 0, 128), (30, 144, 255), (0, 165, 255), (60, 179, 113))


def render_meshes_fast(
    image: np.ndarray,
    vertices_by_key: Dict[str, np.ndarray],
    faces: np.ndarray,
    camera: Camera,
    colors: Sequence[Tuple[int, int, int]] = _DEFAULT_COLORS,
    alpha: float = 0.7,
    supersample: int = 2,
) -> np.ndarray:
    """Overlay one or more meshes (sharing one face topology) onto `image`.

    Args:
        image: BGR image as captured by `camera` (already the as-shipped,
            distorted image -- this projects mesh vertices with `camera`'s
            real distortion model, no undistortion needed).
        vertices_by_key: ``{key: (V,3)}`` world-space mesh vertices, e.g.
            ``{"aria01": smpl_dict["aria01"]["vertices"]}`` or
            ``{"left": mano_dict["left"]["vertices"], "right": ...}``.
        faces: ``(F,3)`` int triangle indices, shared by every mesh in
            `vertices_by_key` (true for SMPL, SMPL-X, and MANO).
        camera: the `Camera` that captured `image`.
        colors: BGR fill color per key, cycled if there are more keys than colors.
        alpha: overlay blend weight (1.0 = fully opaque mesh).
        supersample: render at this integer multiple of `image`'s resolution
            then downsample, for smoother silhouette edges.

    Returns: a new BGR image; `image` is not modified in place.
    """
    height, width = image.shape[:2]
    hr_height, hr_width = height * supersample, width * supersample
    canvas = cv2.resize(image, (hr_width, hr_height), interpolation=cv2.INTER_LINEAR)
    faces = np.asarray(faces)

    for index, (key, vertices) in enumerate(sorted(vertices_by_key.items())):
        color = colors[index % len(colors)]
        points_2d = camera.project(np.asarray(vertices)) * supersample
        polygons = np.asarray(points_2d[faces], dtype=np.int32)
        cv2.fillPoly(canvas, polygons, color=color)
        cv2.polylines(canvas, polygons, isClosed=True, color=(0, 0, 0), thickness=supersample, lineType=cv2.LINE_AA)

    rendered = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(rendered, alpha, image, 1 - alpha, 0)
