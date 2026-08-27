#!/usr/bin/env python3
"""Undistort one or more frames from an exo or Aria camera stream.

Examples:
    # single frame
    python scripts/undistort_images.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --output-dir out/undistorted

    # a whole stream
    python scripts/undistort_images.py \\
        --sequence-path /path/to/001_<name> --camera aria01 --mode left \\
        --output-dir out/undistorted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import io, undistort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--frame", type=int, help="a single frame id; omit to process the whole stream")
    parser.add_argument(
        "--balance", type=float, default=None,
        help="target FOV/crop tradeoff; omit for each camera model's reference-matching default "
             "(see contact4d.undistort._default_target_intrinsics)",
    )
    parser.add_argument("--fov-scale", type=float, default=1.0)
    parser.add_argument(
        "--max-valid-angle-deg", type=float, default=None,
        help="clip target rays beyond this angle (degrees) off the optical axis, to suppress "
             "ghosting artifacts from a miscalibrated camera; unset by default",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    trajectory = io.MetricExtrinsics(args.sequence_path, args.camera, args.mode)
    frame_ids = [args.frame] if args.frame is not None else trajectory.frame_ids.tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in frame_ids:
        camera = trajectory.camera_at(frame_id)
        source_path = io.image_path(args.sequence_path, args.camera, args.mode, frame_id)
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"could not read {source_path}")
        result = undistort.undistort_image(
            image, camera, balance=args.balance, fov_scale=args.fov_scale,
            max_valid_angle_deg=args.max_valid_angle_deg,
        )
        destination = args.output_dir / args.camera / args.mode / f"{frame_id:05d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), result)
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
