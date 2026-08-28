#!/usr/bin/env python3
"""Overlay SMPL / SMPL-X / MANO meshes on an image for one camera/frame.

Example:
    python scripts/visualize_models.py \\
        --sequence-path /path/to/001_<name> --camera cam05 --frame 1 \\
        --models smpl mano --body-models-dir /path/to/body_models \\
        --renderer fast --output-dir out/vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact4d import io
from contact4d.body_models import BodyModelCache
from contact4d.mesh_render_fast import render_meshes_fast
from contact4d.undistort import undistort_image

SOURCE_DIR = {"smpl": "smpl", "smplx": "smplx", "mano": "mano"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence-path", required=True, type=Path)
    parser.add_argument("--camera", required=True, help='e.g. "cam05" or "aria01"')
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--models", nargs="+", choices=sorted(SOURCE_DIR), default=sorted(SOURCE_DIR))
    parser.add_argument("--body-models-dir", required=True, type=Path)
    parser.add_argument("--renderer", choices=["fast", "pyrender"], default="fast")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.renderer == "pyrender":
        from contact4d.mesh_render_pyrender import render_meshes_pyrender as render_meshes
    else:
        render_meshes = render_meshes_fast

    camera = io.load_camera(args.sequence_path, args.camera, args.mode, args.frame)
    models = BodyModelCache(args.body_models_dir)
    image = cv2.imread(str(io.image_path(args.sequence_path, args.camera, args.mode, args.frame)))
    if image is None:
        raise FileNotFoundError("could not read the source image")

    if args.renderer == "pyrender":
        # pyrender is pinhole-only; undistort once, up front, using the
        # default target intrinsics (matches contact4d.mesh_render_pyrender's
        # own IntrinsicsCamera). Doing this once here -- rather than inside
        # render_meshes_pyrender on every call below -- matters because this
        # loop composites multiple model kinds onto one running `image`;
        # undistorting again on each iteration would compound into a badly
        # warped result. See contact4d/mesh_render_pyrender.py's docstring.
        image = undistort_image(image, camera)

    for kind in args.models:
        annotation = io.load_annotation(io.annotation_path(args.sequence_path, SOURCE_DIR[kind], args.frame))
        vertices_by_key = {key: params["vertices"] for key, params in annotation.items()}
        faces = models.faces(kind)
        image = render_meshes(image, vertices_by_key, faces, camera, alpha=args.alpha)

    destination = args.output_dir / args.renderer / args.camera / args.mode / f"{args.frame:05d}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
