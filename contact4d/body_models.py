"""Thin loader around the `smplx` pip package for SMPL / SMPL-X / MANO.

Only needed by `contact4d.camera_space` (to regenerate SMPL/SMPL-X
vertices/joints about the canonical pelvis) and, optionally, by the mesh
visualizers to get triangle face topology -- MANO/SMPL/SMPL-X vertices are
already computed and stored directly in every annotation file, so no
forward pass is required just to draw them.

Body model weight files are gated academic assets and are **not** shipped
with this repository -- see ``docs/body_models.md`` for where to get them
and the exact directory layout expected by `BodyModelCache`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import smplx
import torch

PathLike = Union[str, Path]

_MODEL_FILE = {
    "smpl": ("smpl", "SMPL_NEUTRAL.pkl"),
    "smplx": ("smplx", "SMPLX_NEUTRAL.npz"),
}

BODY_PARAMETER_KEYS = {
    "smpl": ("betas", "body_pose", "global_orient", "transl"),
    "smplx": ("betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"),
}


def _as_batch_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(value).reshape(1, -1), dtype=torch.float32)


class BodyModelCache:
    """Loads SMPL/SMPL-X/MANO once and caches per-shape canonical pelvis positions.

    Expects `body_models_dir` laid out as::

        smpl/SMPL_NEUTRAL.pkl
        smplx/SMPLX_NEUTRAL.npz
        mano/MANO_LEFT.pkl
        mano/MANO_RIGHT.pkl

    Symlink these to wherever you actually placed the downloaded weights if
    you already have them laid out differently.
    """

    def __init__(self, body_models_dir: PathLike):
        self.body_models_dir = Path(body_models_dir)
        self._models: Dict[str, torch.nn.Module] = {}
        self._pelvis_cache: Dict[Tuple[str, bytes], np.ndarray] = {}

    def model(self, kind: str) -> torch.nn.Module:
        """The shared SMPL or SMPL-X (neutral) model."""
        if kind not in ("smpl", "smplx"):
            raise ValueError("use mano_model(side) for MANO")
        if kind not in self._models:
            subdir, filename = _MODEL_FILE[kind]
            model_path = self.body_models_dir / subdir / filename
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"missing {kind} body model weights: {model_path} (see docs/body_models.md)"
                )
            kwargs = {"gender": "neutral"}
            if kind == "smplx":
                kwargs.update(num_betas=10, use_pca=False)
            created = smplx.create(str(model_path), kind, **kwargs).to("cpu")
            created.eval()
            self._models[kind] = created
        return self._models[kind]

    def mano_model(self, side: str) -> torch.nn.Module:
        """The MANO model for one hand side ("left" or "right")."""
        key = f"mano_{side}"
        if key not in self._models:
            mano_dir = self.body_models_dir / "mano"
            weight_path = mano_dir / f"MANO_{side.upper()}.pkl"
            if not weight_path.is_file():
                raise FileNotFoundError(f"missing MANO {side} weights: {weight_path} (see docs/body_models.md)")
            # smplx.create(dir, "mano") appends "mano" to a directory path itself,
            # so it must be pointed at body_models_dir, not body_models_dir/mano.
            created = smplx.create(
                str(self.body_models_dir), "mano", use_pca=False, is_rhand=(side == "right"), flat_hand_mean=True,
            ).to("cpu")
            created.eval()
            self._models[key] = created
        return self._models[key]

    def faces(self, kind: str, side: str = "right") -> np.ndarray:
        """Triangle face topology (fixed per model kind) as an ``(F, 3)`` int array."""
        model = self.mano_model(side) if kind == "mano" else self.model(kind)
        return np.asarray(model.faces)

    def canonical_pelvis(self, kind: str, betas: np.ndarray) -> np.ndarray:
        """Root joint location at zero pose -- the point SMPL/SMPL-X rotate
        about when their global params are re-expressed in camera space."""
        betas = np.asarray(betas, dtype=np.float32).reshape(10)
        key = (kind, betas.tobytes())
        if key not in self._pelvis_cache:
            model = self.model(kind)
            kwargs = {
                "betas": _as_batch_tensor(betas),
                "global_orient": torch.zeros((1, 3), dtype=torch.float32),
                "transl": torch.zeros((1, 3), dtype=torch.float32),
                "body_pose": torch.zeros((1, 69 if kind == "smpl" else 63), dtype=torch.float32),
            }
            if kind == "smplx":
                kwargs.update(
                    left_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
                    right_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
                )
            with torch.no_grad():
                output = model(**kwargs)
            self._pelvis_cache[key] = output.joints[0, 0].detach().cpu().numpy().copy()
        return self._pelvis_cache[key]

    def forward(self, kind: str, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Run the SMPL/SMPL-X forward pass, return ``(vertices, joints)``."""
        kwargs = {key: _as_batch_tensor(params[key]) for key in BODY_PARAMETER_KEYS[kind]}
        with torch.no_grad():
            output = self.model(kind)(**kwargs)
        return (
            output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy(),
        )
