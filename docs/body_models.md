# Body model setup (SMPL / SMPL-X / MANO)

`contact4d.camera_space` (item 2, SMPL/SMPL-X only) and
`scripts/visualize_models.py` (item 3, to read mesh face topology) need the
official SMPL, SMPL-X, and MANO model weights. **These are gated academic
assets and are not shipped with this repository or the dataset** -- you must
request/download them yourself from their official sites:

- SMPL: https://smpl.is.tue.mpg.de
- SMPL-X: https://smpl-x.is.tue.mpg.de
- MANO: https://mano.is.tue.mpg.de

Each requires accepting its own license (non-commercial research use;
commercial use requires contacting the model authors) before download.
**Do not redistribute these weight files.**

## Expected layout

Point every script's `--body-models-dir` (or `contact4d.body_models.BodyModelCache(...)`)
at a directory laid out as:

```
<body_models_dir>/
├── smpl/
│   └── SMPL_NEUTRAL.pkl
├── smplx/
│   └── SMPLX_NEUTRAL.npz
└── mano/
    ├── MANO_LEFT.pkl
    └── MANO_RIGHT.pkl
```

Only the neutral-gender SMPL/SMPL-X models are used (Contact4D's `smpl`/
`smplx` annotations are fit with `gender="neutral"`). If you downloaded the
gendered `.pkl`/`.npz` files instead, rename/symlink the neutral one to the
paths above.

If you already have these models laid out differently (e.g. from an
existing SMPL-X-based project), just symlink each expected path to your
existing files instead of duplicating them:

```bash
mkdir -p body_models/smpl body_models/smplx body_models/mano
ln -s /path/to/SMPL_NEUTRAL.pkl        body_models/smpl/SMPL_NEUTRAL.pkl
ln -s /path/to/SMPLX_NEUTRAL.npz       body_models/smplx/SMPLX_NEUTRAL.npz
ln -s /path/to/MANO_LEFT.pkl           body_models/mano/MANO_LEFT.pkl
ln -s /path/to/MANO_RIGHT.pkl          body_models/mano/MANO_RIGHT.pkl
```

## What doesn't need them

- Item 1 (2D projection), item 4 (undistortion), item 5 (3D keypoints to
  camera space), and MANO's camera-space conversion in item 2 need **no**
  body model weights at all -- they only work with keypoints/rigid
  transforms.
- `scripts/visualize_models.py --renderer fast|pyrender` needs
  `--body-models-dir` only to read each model's fixed triangle face
  topology (`.faces`) -- vertex *positions* come straight from the
  annotation files, no forward pass is run for visualization.
- Only SMPL/SMPL-X camera-space conversion (`contact4d.camera_space.smpl_to_camera_space`
  / `smplx_to_camera_space`) actually runs a body-model forward pass, to
  regenerate vertices/joints about the correct rotation center.
