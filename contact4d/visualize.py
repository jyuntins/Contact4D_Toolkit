"""2D skeleton drawing helpers for the outputs of `contact4d.projection`."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .finger_contact import FINGER_NAMES

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

_CONTACT_COLOR = (60, 60, 220)      # BGR red -- fingertip in contact
_NO_CONTACT_COLOR = (60, 200, 60)   # BGR green -- fingertip not in contact
_PANEL_BG = (25, 25, 25)
_PANEL_TEXT_COLOR = (240, 240, 240)
_FINGER_ABBR = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}


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


def _draw_finger_contact_panel(
    canvas: np.ndarray, frame_contact: Dict[str, bool], side: str, corner: str, scale: float,
) -> None:
    """Draw one hand's 5-finger contact state as a small panel in a corner (in place)."""
    w = canvas.shape[1]
    dot_r = round(16 * scale)
    gap = round(14 * scale)
    pad = round(18 * scale)
    label_h = round(30 * scale)
    n = len(FINGER_NAMES)
    panel_w = pad * 2 + n * (2 * dot_r) + (n - 1) * gap
    panel_h = pad + label_h + 2 * dot_r + pad

    x0 = pad if corner == "left" else w - pad - panel_w
    y0 = pad

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BG, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)

    title = f"{side.upper()} HAND"
    cv2.putText(canvas, title, (x0 + pad, y0 + pad + round(14 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale, _PANEL_TEXT_COLOR, max(1, round(scale)), cv2.LINE_AA)

    cy = y0 + label_h + pad + dot_r
    for i, finger in enumerate(FINGER_NAMES):
        cx = x0 + pad + dot_r + i * (2 * dot_r + gap)
        in_contact = bool(frame_contact.get(f"{side}_{finger}", False))
        color = _CONTACT_COLOR if in_contact else _NO_CONTACT_COLOR
        cv2.circle(canvas, (cx, cy), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), dot_r, (15, 15, 15), max(1, round(2 * scale)), cv2.LINE_AA)
        label = _FINGER_ABBR[finger]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, round(scale)))
        cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (20, 20, 20), max(1, round(scale)), cv2.LINE_AA)


def draw_finger_contact_2d(
    image: np.ndarray,
    frame_contact: Dict[str, bool],
    mirror: bool = True,
    scale: float = 4.0,
) -> np.ndarray:
    """Overlay per-hand fingertip contact state as two corner panels.

    `frame_contact` is one entry from `contact4d.io.load_finger_contact`
    (``{"left_thumb": bool, ..., "right_pinky": bool}``). Draws two fixed
    panels -- one per hand side, five dots each in `FINGER_NAMES` order
    (thumb/index/middle/ring/pinky, labeled T/I/M/R/P), red = in contact,
    green = not in contact -- in the top corners of `image`. This is a HUD,
    not an on-hand overlay: it needs no 2D hand keypoints, and a panel's
    screen position is fixed regardless of where that hand actually is in
    frame.

    `mirror` (default True): put the RIGHT hand's panel in the top-left
    corner and the LEFT hand's panel in the top-right corner. Correct for
    the common case of a subject facing an exo camera, where their right
    hand tends to actually appear on the viewer's left side of frame and
    vice versa. Set `mirror=False` for an egocentric (Aria) view, where
    each hand already appears on its own matching screen side.

    `scale`: linear size multiplier for the panels (dot radius, padding,
    text). Returns a new BGR image; `image` is not modified in place.
    """
    canvas = image.copy()
    left_corner, right_corner = ("right", "left") if mirror else ("left", "right")
    _draw_finger_contact_panel(canvas, frame_contact, "left", corner=left_corner, scale=scale)
    _draw_finger_contact_panel(canvas, frame_contact, "right", corner=right_corner, scale=scale)
    return canvas
