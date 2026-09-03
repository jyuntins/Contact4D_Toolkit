#!/usr/bin/env python3
"""Overlay ground-truth fingertip contact state for one camera/frame.

Draws two corner panels (one per hand, five dots each: thumb/index/middle/
ring/pinky) rather than marking up the hand itself -- no keypoint
projection needed.

Example:
    python scripts/visualize_finger_contact.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --output-dir out/finger_contact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import io, visualize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb", help='"rgb" for exo; "rgb"/"left"/"right" for aria')
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--no-mirror", action="store_false", dest="mirror",
        help="place each panel on its own hand's side instead of mirrored -- use this for an egocentric (Aria) view",
    )
    args = parser.parse_args()

    finger_contact = io.load_finger_contact(args.sequence_path)
    if args.frame not in finger_contact:
        raise KeyError(f"frame {args.frame} has no finger_contact annotation")

    image = cv2.imread(str(io.image_path(args.sequence_path, args.camera, args.mode, args.frame)))
    if image is None:
        raise FileNotFoundError("could not read the source image")

    overlay = visualize.draw_finger_contact_2d(image, finger_contact[args.frame], mirror=args.mirror)
    destination = args.output_dir / args.camera / args.mode / f"{args.frame:05d}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
