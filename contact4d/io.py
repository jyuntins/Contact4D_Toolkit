"""I/O helpers for reading Contact4D annotations and camera trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .cameras import Camera

PathLike = Union[str, Path]


def load_annotation(path: PathLike) -> dict:
    """Load one of Contact4D's per-frame annotation ``.npy`` files.

    These are saved as 0-d numpy object arrays wrapping a plain dict (see
    ``docs/annotations.md``), e.g. ``processed_data/smpl/00001.npy`` ->
    ``{"aria01": {"betas": ..., "vertices": ..., ...}}``.
    """
    path = Path(path)
    value = np.load(str(path), allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        value = value.item()
    if not isinstance(value, dict):
        raise ValueError(f"expected a dict-valued annotation file: {path}")
    return value


def load_finger_contact(sequence_path: PathLike) -> Dict[int, Dict[str, bool]]:
    """Load ``processed_data/finger_contact/annotations.json``.

    Unlike every other annotation type (one ``.npy`` per frame), this is a
    single JSON file per sequence: ``{"<frame_id>": {"left_thumb": bool,
    "left_index": bool, ..., "right_pinky": bool}}``, one entry per
    annotated frame -- not every frame in the sequence necessarily has one.
    Returns the same shape with integer frame-id keys, matching every other
    frame-id convention in this package. See `contact4d.finger_contact` for
    the fingertip-to-MANO-joint mapping used to visualize this.
    """
    path = Path(sequence_path) / "processed_data" / "finger_contact" / "annotations.json"
    if not path.is_file():
        raise FileNotFoundError(f"no finger_contact annotations: {path}")
    raw = json.loads(path.read_text())
    return {int(frame_id): contacts for frame_id, contacts in raw.items()}


def numeric_frame_ids(directory: PathLike, suffix: str = ".npy") -> List[int]:
    """Sorted list of integer frame ids for files named like ``00001<suffix>``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        int(path.stem) for path in directory.glob(f"*{suffix}") if path.stem.isdigit()
    )


def annotation_path(sequence_path: PathLike, kind: str, frame_id: int) -> Path:
    """Path to ``processed_data/<kind>/<frame_id:05d>.npy`` within a sequence."""
    return Path(sequence_path) / "processed_data" / kind / f"{frame_id:05d}.npy"


def image_path(sequence_path: PathLike, camera: str, mode: str, frame_id: int) -> Path:
    """Path to the raw JPEG for one (camera, mode, frame).

    Exo cameras (``camNN``) are stored at ``exo/<camera>/images/<frame>.jpg``
    (only ``mode="rgb"`` exists). Aria cameras are stored at
    ``ego/<camera>/images/<mode>/<frame>.jpg`` with ``mode`` one of
    ``rgb``/``left``/``right``.
    """
    sequence_path = Path(sequence_path)
    filename = f"{frame_id:05d}.jpg"
    if camera.startswith("aria"):
        return sequence_path / "ego" / camera / "images" / mode / filename
    return sequence_path / "exo" / camera / "images" / filename


class MetricExtrinsics:
    """One camera/mode's full ``processed_data/camera_params`` trajectory.

    Falls back to the older ``metric_extrinsics`` folder name if
    ``camera_params`` isn't present in a given sequence.
    """

    def __init__(self, sequence_path: PathLike, camera: str, mode: str = "rgb"):
        self.sequence_path = Path(sequence_path)
        self.camera = camera
        self.mode = mode
        root = self.sequence_path / "processed_data" / "camera_params"
        if not root.is_dir():
            legacy_root = self.sequence_path / "processed_data" / "metric_extrinsics"
            if legacy_root.is_dir():
                root = legacy_root
        npz_path = root / camera / f"{mode}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"no camera_params for {camera}/{mode}: {npz_path}")

        with np.load(str(npz_path), allow_pickle=False) as data:
            self.frame_ids = data["frame_ids"].astype(int)
            self.world_to_camera = data["world_to_camera"]
            self.camera_to_world = data["camera_to_world"]
            self.intrinsics = data["intrinsics"]
            if "source_calibration_frame_ids" in data.files:
                self.source_calibration_frame_ids = data["source_calibration_frame_ids"].astype(int)
            else:
                self.source_calibration_frame_ids = np.full_like(self.frame_ids, -1)
            self.image_width = int(np.asarray(data["image_width"]))
            self.image_height = int(np.asarray(data["image_height"]))
        self._index_by_frame: Dict[int, int] = {
            frame: index for index, frame in enumerate(self.frame_ids.tolist())
        }

        # Fall back to inferring the model from the camera name when this
        # sequence's metadata doesn't carry an explicit camera_model entry.
        self.camera_model = "ARIA_RADTAN_THIN_PRISM_FISHEYE_15" if camera.startswith("aria") else "OPENCV_FISHEYE"
        metadata_path = root / "_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            entry = metadata.get("camera_metadata", {}).get(f"{camera}/{mode}")
            if entry is not None:
                self.camera_model = entry["camera_model"]

    def frame_index(self, frame_id: int) -> int:
        try:
            return self._index_by_frame[frame_id]
        except KeyError:
            raise KeyError(
                f"frame {frame_id} not in {self.camera}/{self.mode} camera_params"
            ) from None

    def camera_at(self, frame_id: int) -> Camera:
        """Build a `Camera` for one frame of this trajectory."""
        index = self.frame_index(frame_id)
        return Camera.from_camera_params_frame(
            name=self.camera,
            mode=self.mode,
            camera_model=self.camera_model,
            world_to_camera=self.world_to_camera[index],
            intrinsics=self.intrinsics[index],
            image_width=self.image_width,
            image_height=self.image_height,
        )


def list_metric_cameras(sequence_path: PathLike) -> List[Tuple[str, str]]:
    """List every ``(camera, mode)`` pair exported to ``camera_params``."""
    root = Path(sequence_path) / "processed_data" / "camera_params"
    if not root.is_dir():
        legacy_root = Path(sequence_path) / "processed_data" / "metric_extrinsics"
        if legacy_root.is_dir():
            root = legacy_root
    metadata_path = root / "_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if "cameras" in metadata:
            return [tuple(entry) for entry in metadata["cameras"]]
    pairs = []
    for camera_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for npz_path in sorted(camera_dir.glob("*.npz")):
            pairs.append((camera_dir.name, npz_path.stem))
    return pairs


def load_camera(sequence_path: PathLike, camera: str, mode: str, frame_id: int) -> Camera:
    """Convenience one-shot: build a `Camera` for a single (camera, mode, frame)."""
    return MetricExtrinsics(sequence_path, camera, mode).camera_at(frame_id)
