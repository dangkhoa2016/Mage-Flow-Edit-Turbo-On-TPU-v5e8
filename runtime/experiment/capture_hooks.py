# -*- coding: utf-8 -*-
"""capture_hooks — C0..C7 pipeline capture plumbing (the core corrective deliverable).

Captures:
  C0 Text Encoder output
  C1 RoPE / positional tensor
  C2 selected Transformer boundary (img stream, txt stream, timesteps, cu_seqlens)
  C3 Transformer final output
  C4 denoised latent (per step + final)
  C5 pre-VAE latent
  C6 VAE output tensor
  C7 final image metadata/artifact

Invariants:
  - explicit run directory, deterministic checkpoint names
  - metadata written atomically (tmp file + rename)
  - never modifies model parameters
  - never alters tensors passed downstream
  - distinguishes host copies from device tensors
  - never silently converts the compute path dtype
  - importable and instantiable WITHOUT any TPU or model
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import tensor_metrics as tm


def _atomic_write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _host_copy(arr: Any, dtype: Optional[str] = None) -> Any:
    """Return a host-side numpy view/copy of a tensor WITHOUT guiding compute to CPU."""
    if hasattr(arr, "numpy") and not hasattr(arr, "device"):
        arr = arr.numpy()
    import numpy as np

    a = np.asarray(arr)
    if dtype is not None and str(a.dtype) != dtype:
        raise AssertionError(
            f"capture dtype mismatch requested={dtype} actual={a.dtype}; refusing silent cast"
        )
    return a


@dataclass
class Capture:
    """One captured checkpoint."""

    name: str
    run_dir: Path

    @property
    def cp_name(self) -> str:
        return f"{self.name}.json"

    @property
    def artifact_name(self) -> str:
        return f"{self.name}.npy"

    def store_tensor(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        """Persist a host copy as a deterministic checkpoint (never mutates the tensor)."""
        run = Path(self.run_dir)
        run.mkdir(parents=True, exist_ok=True)
        host = _host_copy(tensor)
        rec = tm.tensor_record(self.name, host)
        rec["host_copy"] = True
        device = getattr(tensor, "device", None)
        rec["device_tensor_was"] = None if device is None else str(device)
        if extra:
            rec["extra"] = extra
        meta_p = run / self.cp_name
        _atomic_write_json(meta_p, rec)
        np = __import__("numpy")
        np.save(run / self.artifact_name, host)
        return meta_p

    def store_metadata(self, payload: dict) -> Path:
        run = Path(self.run_dir)
        run.mkdir(parents=True, exist_ok=True)
        return _atomic_write_json(run / self.cp_name, payload)


class CaptureHooks:
    """C0-C7 capture hook set. Importable/instantiable without TPU.

    A hook may copy a small diagnostic representation to host for persistence; it must not
    cause full main-model execution to move to CPU, and it never alters tensors downstream.
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)

    def c0_text_encoder_output(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        return Capture("C0_text_encoder_output", self.run_dir).store_tensor(tensor, extra)

    def c1_rope(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        return Capture("C1_rope", self.run_dir).store_tensor(tensor, extra)

    def c2_transformer_boundary(self, payload: dict) -> Path:
        return Capture("C2_transformer_boundary", self.run_dir).store_metadata(payload)

    def c2_transformer_boundary_tensor(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        """Persist the actual C2 tensor (not metadata only) with its boundary description."""
        return Capture("C2_transformer_boundary", self.run_dir).store_tensor(tensor, extra)

    def c3_transformer_final_output(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        return Capture("C3_transformer_final_output", self.run_dir).store_tensor(tensor, extra)

    def c4_denoised_latent(self, tensor: Any, step: int, extra: Optional[dict] = None) -> Path:
        cap = Capture(f"C4_denoised_latent_step{step}", self.run_dir)
        return cap.store_tensor(
            tensor,
            {"step": step, **({"extra": extra} if extra else {})},
        )

    def c4_final_latent(self, tensor: Any) -> Path:
        return Capture("C4_final_denoised_latent", self.run_dir).store_tensor(tensor)

    def c5_pre_vae_latent(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        return Capture("C5_pre_vae_latent", self.run_dir).store_tensor(tensor, extra)

    def c6_vae_output(self, tensor: Any, extra: Optional[dict] = None) -> Path:
        return Capture("C6_vae_output", self.run_dir).store_tensor(tensor, extra)

    def c7_final_image(self, png_bytes: bytes, pixel_stats: dict, png_sha256: str) -> Path:
        run = Path(self.run_dir)
        run.mkdir(parents=True, exist_ok=True)
        png_p = run / "final.png"
        with open(png_p, "wb") as fh:
            fh.write(png_bytes)
        (run / "final.png.sha256").write_text(png_sha256 + "\n")
        payload = {
            "name": "C7_final_image",
            "image_png": str(png_p),
            "png_sha256": png_sha256,
            "pixel_stats": pixel_stats,
        }
        return _atomic_write_json(run / "C7_final_image.json", payload)


def hook_smoke_test(tmpdir: str) -> dict:
    """Tiny synthetic interface test of every C0-C7 hook (CPU, no model, no TPU)."""
    import numpy as np

    hooks = CaptureHooks(tmpdir)
    t = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    results = {
        "C0": str(hooks.c0_text_encoder_output(t)),
        "C1": str(hooks.c1_rope(t)),
        "C2": str(hooks.c2_transformer_boundary({"img": [1, 128]})),
        "C3": str(hooks.c3_transformer_final_output(t)),
        "C4_step": str(hooks.c4_denoised_latent(t, step=2)),
        "C4_final": str(hooks.c4_final_latent(t)),
        "C5": str(hooks.c5_pre_vae_latent(t)),
        "C6": str(hooks.c6_vae_output(t)),
        "C7": str(hooks.c7_final_image(b"png-bytes", {"mean": 1.0}, "aa" * 32)),
    }
    for k, v in results.items():
        p = Path(v)
        assert p.exists(), f"missing capture artifact {p}"
    return results


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        res = hook_smoke_test(td)
        print("CAPTURE_HOOKS_FILES", res)
    print("CAPTURE_HOOKS_SMOKE=PASS")