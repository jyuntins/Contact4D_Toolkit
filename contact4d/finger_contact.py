"""Ground-truth fingertip contact annotations (item 6).

``processed_data/finger_contact/annotations.json`` (loaded via
`contact4d.io.load_finger_contact`) gives one ``{"left_thumb": bool,
"left_index": bool, ..., "right_pinky": bool}`` dict per annotated frame:
whether each fingertip is in contact with an object/surface in that frame.

This maps directly onto the 21-joint MANO hand skeleton used throughout
this package (`contact4d.visualize.HAND_EDGES`, `hand_pose3d`, MANO
itself): each finger's tip is the last joint in its chain -- `FINGERTIP_JOINTS`
below gives that mapping, for callers that need a fingertip's actual 3D/2D
position (e.g. to overlay contact state directly on the hand rather than
via `contact4d.visualize.draw_finger_contact_2d`'s corner panels, which
don't need it).
"""

from __future__ import annotations

from typing import Dict

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# MANO joint index of each fingertip -- the last joint in each finger's
# chain in contact4d.visualize.HAND_EDGES.
FINGERTIP_JOINTS: Dict[str, int] = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def contact_by_finger(frame_contact: Dict[str, bool], side: str) -> Dict[str, bool]:
    """Extract one hand side's 5 per-finger contact booleans.

    `frame_contact` is one entry from `contact4d.io.load_finger_contact`
    (e.g. ``finger_contact[frame_id]``); `side` is ``"left"`` or ``"right"``.
    """
    return {finger: bool(frame_contact[f"{side}_{finger}"]) for finger in FINGER_NAMES}
