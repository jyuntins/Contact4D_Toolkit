# Contact4D toolkit

Official project website: [https://jyuntins.github.io/Contact4D/](https://jyuntins.github.io/Contact4D/)

**[Website](https://jyuntins.github.io/Contact4D/) · [Paper](https://openreview.net/pdf?id=5DPvfQtAjm) · [Dataset](https://huggingface.co/datasets/Jyun-Ting/Contact4D/tree/main)**

Contact4D is a large-scale egocentric+exocentric human motion capture
dataset: synchronized footage from ~18 fixed exo cameras and one or more
Aria egocentric devices per sequence, with per-frame 3D body/hand
keypoints, fitted SMPL/SMPL-X/MANO body and hand models, and full camera
calibration -- all in one shared metric world frame.

This repository is the **Contact4D dataset toolkit** (consumer side): everything
you need to turn the released annotations into 2D projections, camera-space
model parameters, mesh visualizations, and undistorted images, without
needing any of the internal capture/processing pipeline. It's a standalone,
lightweight package (`numpy`/`opencv`/`scipy`/`torch`/`smplx`) -- no COLMAP,
no proprietary config system, and (despite the egocentric data including
Aria cameras) no Aria SDK dependency either: both camera models are
re-implemented from scratch in plain numpy.

## Install

```bash
git clone <this-repo-url> contact4d
cd contact4d
pip install -r requirements.txt
pip install -e .          # optional: makes `import contact4d` work from anywhere
```

See [`docs/installation.md`](docs/installation.md) for a virtual environment
setup, GPU/torch notes, verifying the install, and the optional
high-quality (`pyrender`) renderer setup.

## What's here

| # | Capability | Script | Package |
|---|---|---|---|
| 1 | Project 3D body/hand keypoints to 2D | `scripts/project_pose3d_to_2d.py` | `contact4d.projection` |
| 2 | Convert global SMPL/SMPL-X/MANO to camera space | `scripts/convert_to_camera_space.py` | `contact4d.camera_space` |
| 3 | Visualize SMPL/SMPL-X/MANO on images | `scripts/visualize_models.py` | `contact4d.mesh_render_fast`, `contact4d.mesh_render_pyrender` |
| 4 | Undistort images (exo or Aria) | `scripts/undistort_images.py` | `contact4d.undistort` |
| 5 | 3D keypoints to camera space | `scripts/keypoints_to_camera_space.py` | `contact4d.keypoints_camera_space` |
| 6 | Visualize ground-truth fingertip contact | `scripts/visualize_finger_contact.py` | `contact4d.finger_contact` |

Every script takes `--sequence-path` pointing at one downloaded sequence
directory (e.g. `.../001_<name>/`) and reads calibration straight from its
`processed_data/camera_params/`.

### Quick examples

```bash
SEQ=/path/to/001_<name>

# 1. project body_pose3d + hand keypoints to cam05, and draw them
python scripts/project_pose3d_to_2d.py --sequence-path "$SEQ" --camera cam05 --frame 1 \
    --output-dir out/pose2d --visualize

# 2. convert MANO to cam05's camera space (no body models needed)
python scripts/convert_to_camera_space.py --sequence-path "$SEQ" --camera cam05 --frame 1 \
    --models mano --output-dir out/camera_space

# 3. overlay SMPL-X + MANO meshes on cam05 (needs body models, see docs/body_models.md)
python scripts/visualize_models.py --sequence-path "$SEQ" --camera cam05 --frame 1 \
    --models smplx mano --body-models-dir /path/to/body_models --output-dir out/vis

# 4. undistort an Aria left (SLAM) stream
python scripts/undistort_images.py --sequence-path "$SEQ" --camera aria01 --mode left \
    --output-dir out/undistorted

# 5. bring body_pose3d into aria01/rgb's camera frame
python scripts/keypoints_to_camera_space.py --sequence-path "$SEQ" --camera aria01 --mode rgb \
    --frame 1 --source body_pose3d --output-dir out/keypoints_cam

# 6. overlay ground-truth fingertip contact state on cam05
python scripts/visualize_finger_contact.py --sequence-path "$SEQ" --camera cam05 --frame 1 \
    --output-dir out/finger_contact
```

## Docs

- [`docs/installation.md`](docs/installation.md) -- setup, GPU notes, verifying the install, optional high-quality rendering.
- [`docs/data_structure.md`](docs/data_structure.md) -- directory layout and naming conventions.
- [`docs/annotations.md`](docs/annotations.md) -- exact schema for every annotation type.
- [`docs/cameras.md`](docs/cameras.md) -- calibration format and both camera (distortion) models.
- [`docs/body_models.md`](docs/body_models.md) -- where to get SMPL/SMPL-X/MANO and how to point the toolkit at them.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

The test suite is synthetic-data-only (no dataset download required) and
checks the ported camera math and camera-space conversion against
independently-computed ground truth.

## License

The code in this repository is MIT-licensed (see `LICENSE`). The Contact4D
dataset itself and the third-party SMPL/SMPL-X/MANO body models are under
their own separate terms -- see `LICENSE` and `docs/body_models.md`.
