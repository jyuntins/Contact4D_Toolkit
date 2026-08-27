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
    """One camera/mode's full ``processed_data/metric_extrinsics`` trajectory."""

    def __init__(self, sequence_path: PathLike, camera: str, mode: str = "rgb"):
        self.sequence_path = Path(sequence_path)
        self.camera = camera
        self.mode = mode
        root = self.sequence_path / "processed_data" / "metric_extrinsics"
        npz_path = root / camera / f"{mode}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"no metric_extrinsics for {camera}/{mode}: {npz_path}")

        with np.load(str(npz_path), allow_pickle=False) as data:
            self.frame_ids = data["frame_ids"].astype(int)
            self.world_to_camera = data["world_to_camera"]
            self.camera_to_world = data["camera_to_world"]
            self.intrinsics = data["intrinsics"]
            self.source_calibration_frame_ids = data["source_calibration_frame_ids"].astype(int)
            self.image_width = int(np.asarray(data["image_width"]))
            self.image_height = int(np.asarray(data["image_height"]))
        self._index_by_frame: Dict[int, int] = {
            frame: index for index, frame in enumerate(self.frame_ids.tolist())
        }

        self.camera_model = "OPENCV_FISHEYE"
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
                f"frame {frame_id} not in {self.camera}/{self.mode} metric_extrinsics"
            ) from None

    def camera_at(self, frame_id: int) -> Camera:
        """Build a `Camera` for one frame of this trajectory."""
        index = self.frame_index(frame_id)
        return Camera.from_metric_extrinsics_frame(
            name=self.camera,
            mode=self.mode,
            camera_model=self.camera_model,
            world_to_camera=self.world_to_camera[index],
            intrinsics=self.intrinsics[index],
            image_width=self.image_width,
            image_height=self.image_height,
        )


def list_metric_cameras(sequence_path: PathLike) -> List[Tuple[str, str]]:
    """List every ``(camera, mode)`` pair exported to ``metric_extrinsics``."""
    root = Path(sequence_path) / "processed_data" / "metric_extrinsics"
    metadata_path = root / "_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        return [tuple(entry) for entry in metadata["cameras"]]
    pairs = []
    for camera_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for npz_path in sorted(camera_dir.glob("*.npz")):
            pairs.append((camera_dir.name, npz_path.stem))
    return pairs


def load_camera(sequence_path: PathLike, camera: str, mode: str, frame_id: int) -> Camera:
    """Convenience one-shot: build a `Camera` for a single (camera, mode, frame)."""
    return MetricExtrinsics(sequence_path, camera, mode).camera_at(frame_id)
