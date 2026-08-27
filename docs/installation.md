# Installation

## Requirements

- Python >= 3.9
- `numpy`, `opencv-python`, `scipy`, `torch`, `smplx` (installed below) -- no
  COLMAP, no Aria SDK, no GPU required for items 1, 2 (MANO path), 4, 5.
  Items 2 (SMPL/SMPL-X path) and 3 use `torch`/`smplx` for a body-model
  forward pass and run fine on CPU; a GPU only helps if you're processing
  many frames.
- Optionally `pyrender` + `trimesh` for the high-quality mesh renderer
  (`scripts/visualize_models.py --renderer pyrender`) -- see
  [Optional: high-quality rendering](#optional-high-quality-rendering) below.
- Separately, the SMPL/SMPL-X/MANO body model weight files if you'll use
  item 2 (SMPL/SMPL-X) or item 3 (visualization) -- see
  [`docs/body_models.md`](body_models.md). Nothing else in this repo needs
  them.

## Basic install

A virtual environment (`venv` or `conda`) is recommended so this doesn't
collide with other projects' package versions:

```bash
python3 -m venv .venv
source .venv/bin/activate   # or: conda create -n contact4d python=3.11 && conda activate contact4d
```

Then clone and install:

```bash
git clone <this-repo-url> contact4d
cd contact4d
pip install -r requirements.txt
pip install -e .              # makes `import contact4d` work from anywhere, and installs the CLI scripts as modules
```

`torch` installs a CPU build by default via the plain `pip install`. If you
want GPU acceleration, install the CUDA build matching your driver *before*
`pip install -r requirements.txt` (so the later install sees it's already
satisfied) -- see https://pytorch.org/get-started/locally/ for the exact
command for your CUDA version.

## Verify it worked

```bash
pip install -e ".[dev]"
pytest tests/
```

This test suite is synthetic-data-only (no dataset download needed) and
checks the ported camera-projection and camera-space-conversion math
directly. All tests should pass with just the basic install above --
`pyrender`/`trimesh` and body model weights are not required for the tests.

To check the actual CLI scripts against real data, follow the quick
examples in the top-level [`README.md`](../README.md) against a downloaded
sequence directory.

## Optional: high-quality rendering

```bash
pip install -e ".[render]"    # adds pyrender + trimesh
```

`scripts/visualize_models.py`'s default `--renderer fast` (flat-shaded,
pure OpenCV) needs nothing beyond the basic install and works everywhere.
`--renderer pyrender` needs a working headless-OpenGL setup:

- On a machine with an NVIDIA GPU, install EGL support (usually already
  present with recent NVIDIA drivers) -- no display/X server needed.
- Without a GPU, install OSMesa instead and set
  `PYOPENGL_PLATFORM=osmesa` before running (`pyrender`'s software
  rasterizer; slower, but works on any machine).
- If `pyrender.OffscreenRenderer(...)` raises an OpenGL/EGL error, that's
  almost always a missing system-level OpenGL/EGL library, not a Python
  package problem -- the fast renderer remains a full-featured fallback in
  the meantime.

## Body model weights (SMPL / SMPL-X / MANO)

Not installed by pip -- these are gated academic assets you download
yourself and point `--body-models-dir` at. See
[`docs/body_models.md`](body_models.md) for the download links, license
terms, and expected directory layout.
