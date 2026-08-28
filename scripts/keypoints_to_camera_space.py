#!/usr/bin/env python3
"""Transform 3D body/hand keypoints from world space into one camera's frame.

Example:
    python scripts/keypoints_to_camera_space.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --source body_pose3d --output-dir out/keypoints_cam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import io, keypoints_camera_space


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument(
        "--source", default="body_pose3d",
        choices=["body_pose3d", "hand_pose3d"],
        help="which world-space keypoint annotation to transform",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    camera = io.load_camera(args.sequence_path, args.camera, args.mode, args.frame)
    keypoints_world = io.load_annotation(io.annotation_path(args.sequence_path, args.source, args.frame))
    keypoints_cam = keypoints_camera_space.transform_keypoints_to_camera(keypoints_world, camera)

    if args.output_dir:
        destination = args.output_dir / args.source / args.camera / args.mode / f"{args.frame:05d}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, keypoints_cam, allow_pickle=True)
        print(f"wrote {destination}")
    else:
        print(keypoints_cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
