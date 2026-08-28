#!/usr/bin/env python3
"""Convert global SMPL / SMPL-X / MANO annotations to camera space.

Example:
    python scripts/convert_to_camera_space.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --models smpl smplx mano --body-models-dir /path/to/body_models \\
        --output-dir out/camera_space

MANO does not need `--body-models-dir` (its camera-space conversion is a
direct geometric transform, no body-model forward pass); SMPL/SMPL-X do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import camera_space, io
from contact4d.body_models import BodyModelCache

SOURCE_DIR = {"smpl": "smpl", "smplx": "smplx", "mano": "mano"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--models", nargs="+", choices=sorted(SOURCE_DIR), default=sorted(SOURCE_DIR))
    parser.add_argument("--body-models-dir", type=Path, help="required for --models smpl/smplx, see docs/body_models.md")
    parser.add_argument("--tolerance-m", type=float, default=camera_space.DEFAULT_TOLERANCE_M)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if ("smpl" in args.models or "smplx" in args.models) and args.body_models_dir is None:
        parser.error("--body-models-dir is required when converting smpl/smplx")

    camera = io.load_camera(args.sequence_path, args.camera, args.mode, args.frame)
    models = BodyModelCache(args.body_models_dir) if args.body_models_dir else None

    for kind in args.models:
        source = io.load_annotation(io.annotation_path(args.sequence_path, SOURCE_DIR[kind], args.frame))
        if kind == "smpl":
            output, report = camera_space.smpl_to_camera_space(source, camera.world_to_camera, models, args.tolerance_m)
        elif kind == "smplx":
            output, report = camera_space.smplx_to_camera_space(source, camera.world_to_camera, models, args.tolerance_m)
        else:
            output, report = camera_space.mano_to_camera_space(source, camera.world_to_camera, args.tolerance_m)
        print(f"{kind}: {report}")

        if args.output_dir:
            destination = args.output_dir / f"cam_{kind}" / args.camera / args.mode / f"{args.frame:05d}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, output, allow_pickle=True)
            print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
