# Data structure

A Contact4D release is organized as:

```
<super_sequence>/
├── calibration_<super_sequence>/       # shared calibration capture for this super-sequence
│   ├── exo/camNN/images/               # calibration frames per exo camera
│   ├── ego/ariaNN/                     # calibration frames per Aria device
│   └── colmap/workplace/               # COLMAP reconstruction + Aria<->COLMAP alignment
├── 001_<super_sequence>/               # one recorded sequence ("clip")
│   ├── exo/
│   │   ├── cam01/images/00001.jpg ...  # 5-digit, zero-padded, 1-indexed frame numbers
│   │   ├── cam02/images/...
│   │   └── ...                         # up to ~17-18 fixed exo cameras
│   ├── ego/
│   │   └── aria01/
│   │       ├── images/{rgb,left,right}/00001.jpg ...
│   │       └── calib/00001.txt ...     # per-frame Aria calibration (rarely needed directly; use metric_extrinsics instead)
│   ├── colmap/                         # per-sequence COLMAP artifacts
│   └── processed_data/                 # all derived annotations, see docs/annotations.md
├── 002_<super_sequence>/
└── ...
```

## Naming conventions

- **Super-sequence** directory names are either date-based (`20260116_1`, i.e. `<YYYYMMDD>_<capture index>`) or a legacy `<index>_<name>` form (e.g. `42_weancan3`). Directories starting with `calibration_` hold shared calibration data, not a recorded sequence.
- **Sequence** directory names are always `NNN_<super_sequence-derived-name>` (e.g. `001_20260116_4`, `001_weancan3`) -- a zero-padded 3-digit index plus the name portion after the super-sequence's own numeric prefix (if any). This name (not the on-disk super-sequence folder name) is what every per-sequence config, e.g. `configs/<name>/<sequence>.yaml`, and every `metric_extrinsics`/annotation file's implicit sequence identity is keyed on.
- **Frames** are 1-indexed, 5-digit zero-padded (`00001.jpg`, `00001.npy`, ...). `exo/cam01/images/` is the canonical frame-count reference: every other camera stream and every `processed_data/*` annotation folder for a *complete* sequence has exactly the same set of frame ids.
- **Exo cameras** are named `camNN` (`cam01`-`cam18` typically, though `INVALID_EXOS` in a sequence's config can exclude specific cameras for a given capture). Only one mode exists for exo cameras: `rgb`.
- **Aria (ego) devices** are named `ariaNN` (usually just `aria01`). Each has three image streams/`mode`s: `rgb` (square, e.g. 1408x1408), `left`, `right` (SLAM cameras, non-square, e.g. 480x640).

## `processed_data/`

Everything derived from the raw images -- 2D/3D keypoints, fitted body/hand models, camera calibration -- lives under a sequence's `processed_data/` directory, one subfolder per annotation type, one file per frame (or, for `metric_extrinsics`, one file per camera/mode covering the whole sequence). See `docs/annotations.md` for exact schemas and `docs/cameras.md` for the camera/calibration format.
