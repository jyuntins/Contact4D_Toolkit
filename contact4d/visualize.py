"""2D skeleton drawing helpers for the outputs of `contact4d.projection`."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

BODY_EDGES: Sequence[Tuple[int, int]] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)  # 17-joint COCO-style body skeleton, matches body_pose3d joint order

HAND_EDGES: Sequence[Tuple[int, int]] = tuple(
    (a, b)
    for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for a, b in zip(finger[:-1], finger[1:])
)  # 21-joint MANO-order hand skeleton, matches hand_pose3d joint order

_PALETTE = ((30, 220, 30), (40, 40, 240), (240, 160, 30), (220, 40, 220))


def draw_keypoints_2d(
    image: np.ndarray,
    keypoints_2d: Dict[str, np.ndarray],
    kind: str = "body",
    confidence_threshold: float = 0.0,
    radius: int = 4,
    thickness: int = 2,
    label: bool = True,
) -> np.ndarray:
    """Draw a `{key: (N,3)}` (x, y, confidence) 2D keypoint dict on `image`.

    `keypoints_2d` is the output of `contact4d.projection.project_body_pose3d`
    / `project_hand_pose3d`. `kind` selects the skeleton edges ("body" ->
    `BODY_EDGES`, "hand" -> `HAND_EDGES`). Returns a new BGR image; `image`
    is not modified in place.

    In addition to `confidence_threshold`, a point is only drawn if it
    falls within a generous (50%-of-frame) margin around `image`'s bounds.
    This is a backstop, not the primary fix: `project_body_pose3d`/
    `project_hand_pose3d` already zero the confidence of Aria joints beyond
    `contact4d.projection.DEFAULT_ARIA_MAX_VALID_ANGLE_DEG`, which is what
    actually avoids projecting a wearer's own near-camera joints (e.g. an
    eye a few centimeters from a head-mounted Aria device) to a wildly
    out-of-frame pixel coordinate in the first place. This bounds check just
    means a stray far-out-of-frame point -- however it got that way -- draws
    as absent rather than as a line shooting off to some distant corner.
    """
    edges = BODY_EDGES if kind == "body" else HAND_EDGES
    canvas = image.copy()
    height, width = image.shape[:2]
    margin_x, margin_y = width / 2.0, height / 2.0
    for index, (key, keypoints) in enumerate(sorted(keypoints_2d.items())):
        keypoints = np.asarray(keypoints)
        color = _PALETTE[index % len(_PALETTE)]
        in_bounds = (
            (keypoints[:, 0] >= -margin_x) & (keypoints[:, 0] <= width + margin_x)
            & (keypoints[:, 1] >= -margin_y) & (keypoints[:, 1] <= height + margin_y)
        )
        valid = (keypoints[:, 2] > confidence_threshold) & in_bounds
        for a, b in edges:
            if valid[a] and valid[b]:
                point_a = tuple(np.rint(keypoints[a, :2]).astype(int))
                point_b = tuple(np.rint(keypoints[b, :2]).astype(int))
                cv2.line(canvas, point_a, point_b, color, thickness, cv2.LINE_AA)
        for point, is_valid in zip(keypoints[:, :2], valid):
            if is_valid:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), radius, color, -1, cv2.LINE_AA)
        if label and valid.any():
            origin = tuple(np.rint(keypoints[valid][0, :2]).astype(int))
            cv2.putText(canvas, key, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas
