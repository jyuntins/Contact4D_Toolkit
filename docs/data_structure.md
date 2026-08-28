# Data structure

A Contact4D release is organized as:

```
<super_sequence>/                       # e.g. session_01
├── 001_<super_sequence>/               # one recorded sequence ("clip"), e.g. 001_session_01
│   ├── exo/
│   │   ├── cam01/images/00001.jpg ...  # 5-digit, zero-padded, 1-indexed frame numbers
│   │   ├── cam02/images/...
│   │   └── ...                         # up to ~17-18 fixed exo cameras
│   ├── ego/
│   │   └── aria01/
│   │       ├── images/{rgb,left,right}/00001.jpg ...
│   │       └── calib/00001.txt ...     # per-frame Aria calibration (rarely needed directly; use camera_params instead)
│   ├── colmap/                         # per-sequence COLMAP artifacts
│   └── processed_data/                 # all derived annotations, see docs/annotations.md
├── 002_<super_sequence>/
└── ...
```

## Naming conventions

- **Super-sequence** directory names are `session_NN` (e.g. `session_01`).
- **Sequence** directory names are always `NNN_<super_sequence>` (e.g. `001_session_01`) -- a zero-padded 3-digit index plus the super-sequence name. Every `camera_params`/annotation file's implicit sequence identity is keyed on this name.
- **Frames** are 1-indexed, 5-digit zero-padded (`00001.jpg`, `00001.npy`, ...). `exo/cam01/images/` is the canonical frame-count reference: every other camera stream and every `processed_data/*` annotation folder for a *complete* sequence has exactly the same set of frame ids.
- **Exo cameras** are named `camNN` (`cam01`-`cam18` typically -- not every sequence necessarily uses every camera index). Only one mode exists for exo cameras: `rgb`.
- **Aria (ego) devices** are named `ariaNN` (usually just `aria01`). Each has three image streams/`mode`s: `rgb` (square, e.g. 1408x1408), `left`, `right` (SLAM cameras, non-square, e.g. 480x640).

## `processed_data/`

Everything derived from the raw images -- 2D/3D keypoints, fitted body/hand models, camera calibration -- lives under a sequence's `processed_data/` directory, one subfolder per annotation type, one file per frame (or, for `camera_params`, one file per camera/mode covering the whole sequence). See `docs/annotations.md` for exact schemas and `docs/cameras.md` for the camera/calibration format.
