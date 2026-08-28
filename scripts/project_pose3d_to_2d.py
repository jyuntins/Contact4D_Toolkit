#!/usr/bin/env python3
"""Project 3D body/hand keypoints to 2D for one camera/frame.

Example:
    python scripts/project_pose3d_to_2d.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --output-dir out/pose2d --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import io, projection, visualize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb", help='"rgb" for exo; "rgb"/"left"/"right" for aria')
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--kind", choices=["body", "hand", "both"], default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visualize", action="store_true", help="also save a skeleton-overlay JPEG")
    args = parser.parse_args()

    camera = io.load_camera(args.sequence_path, args.camera, args.mode, args.frame)
    results = {}
    if args.kind in ("body", "both"):
        body_3d = io.load_annotation(io.annotation_path(args.sequence_path, "body_pose3d", args.frame))
        results["body"] = projection.project_body_pose3d(body_3d, camera)
    if args.kind in ("hand", "both"):
        hand_3d = io.load_annotation(io.annotation_path(args.sequence_path, "hand_pose3d", args.frame))
        results["hand"] = projection.project_hand_pose3d(hand_3d, camera)

    if args.output_dir:
        for name, keypoints_2d in results.items():
            destination = args.output_dir / name / args.camera / args.mode / f"{args.frame:05d}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, keypoints_2d, allow_pickle=True)
            print(f"wrote {destination}")

    if args.visualize:
        image = cv2.imread(str(io.image_path(args.sequence_path, args.camera, args.mode, args.frame)))
        if image is None:
            raise FileNotFoundError("could not read the source image for --visualize")
        for kind, keypoints_2d in results.items():
            image = visualize.draw_keypoints_2d(image, keypoints_2d, kind=kind)
        destination = (args.output_dir or Path(".")) / "vis" / args.camera / args.mode / f"{args.frame:05d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), image)
        print(f"wrote {destination}")

    if not args.output_dir and not args.visualize:
        print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
