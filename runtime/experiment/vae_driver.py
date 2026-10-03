# -*- coding: utf-8 -*-
"""vae_driver — restore / map / bind / decode for the durable VAE JAX runtime.

C1R GAP-4 (VAE restore+binding) resolution. The prior C1 driver loaded an EXTERNAL plugin
module via ``vae_runtime_module`` and never restored or mapped the converted checkpoint. This
module now:

  * loads the durable VAE runtime source that ships IN this package
    (``experiment/vae_runtime.py`` == the qualified ``mage_vae_jax_runtime.py``);
  * restores the converted VAE checkpoint (839 leaves; flat ``vae/...`` tree);
  * maps the 728 runtime-required leaves via VAE_RUNTIME_BINDING_MANIFEST.json
    (``restored[record.target] -> p[record.source_key]``);
  * binds them into the runtime's ``p`` argument;
  * decodes with the bound params (``decode(p, latent)``, t=0 zero-noise, IDENTITY scaling).

Authority: VAE_RUNTIME_BINDING_MANIFEST.json (coverage 728/728, status CLOSED_PASS),
VAE_SINGLE_TPU_RUNTIME_CLOSEOUT_AUTHORITY.json, VAE3B/VAE3C mapping authorities.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .capture_hooks import CaptureHooks

VAE_DTYPE = "bfloat16"
VAE_FRAMEWORK = "jax"
VAE_SCALING_RULE = "IDENTITY"
VAE_FULL_LEAF_COUNT = 839
VAE_RUNTIME_LEAF_COUNT = 728
VAE_CHECKPOINT_AUTHORITY = {
    "file_count": 12,
    "total_bytes": 272360711,
    "aggregate_sha256": "ec317ea690c9c91701586f77ffc68af026bdf9b2664364d48d339142e0e312f1",
}
DEFAULT_VAE_MANIFEST_HINT = str(Path(__file__).resolve().parents[2] / "manifests" / "VAE_RUNTIME_BINDING_MANIFEST.json")


@dataclass
class VaeRuntime:
    checkpoint_path: Optional[str] = None
    manifest_path: Optional[str] = None
    dtype: str = VAE_DTYPE
    runtime_framework: str = VAE_FRAMEWORK
    scaling_rule: str = VAE_SCALING_RULE
    sharding_contract: dict = field(
        default_factory=lambda: {
            "full_converted_leaf_count": VAE_FULL_LEAF_COUNT,
            "runtime_leaves": VAE_RUNTIME_LEAF_COUNT,
            "array_device_counts": {"8": VAE_RUNTIME_LEAF_COUNT},
            "mesh": "TPU_V5E_8",
        }
    )
    decode_fn: Optional[Any] = None
    params: Any = None
    manifest: Optional[dict] = None
    bound: bool = False

    def assert_contract(self) -> None:
        if self.scaling_rule != VAE_SCALING_RULE:
            raise AssertionError("VAE_SCALING_RULE must be IDENTITY (no 1/sigma scalar)")
        if self.dtype != VAE_DTYPE:
            raise AssertionError("VAE dtype contract violated")


# ---------------------------------------------------------------------------
# Source / manifest / restore / map / bind
# ---------------------------------------------------------------------------
def load_vae_runtime():
    """Lazily import the in-package durable VAE runtime (imports jax)."""
    return importlib.import_module("experiment.vae_runtime")


def load_binding_manifest(manifest_path: str | Path) -> dict:
    d = json.loads(Path(manifest_path).read_text())
    if d.get("status") != "CLOSED_PASS":
        raise AssertionError(f"VAE binding manifest status {d.get('status')!r} != CLOSED_PASS")
    if d.get("runtime_binding_coverage") != f"{VAE_RUNTIME_LEAF_COUNT}/{VAE_RUNTIME_LEAF_COUNT}":
        raise AssertionError(
            f"VAE runtime_binding_coverage {d.get('runtime_binding_coverage')!r} != 728/728"
        )
    if int(d.get("full_converted_leaf_count", -1)) != VAE_FULL_LEAF_COUNT:
        raise AssertionError("VAE manifest full_converted_leaf_count != 839")
    if len(d.get("records", [])) != VAE_RUNTIME_LEAF_COUNT:
        raise AssertionError("VAE manifest record count != 728")
    return d


def _flatten_state(tree: Any) -> dict:
    if isinstance(tree, dict) and tree and all(isinstance(k, str) for k in tree):
        return dict(tree)
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    out: dict = {}
    for path, leaf in flat:
        parts = []
        for entry in path:
            name = getattr(entry, "key", None) or getattr(entry, "name", None) or str(entry)
            parts.append(str(name))
        out["/".join(parts)] = leaf
    return out


def restore_vae_checkpoint(
    checkpoint_path: str | Path, *, expected_leaf_count: int = VAE_FULL_LEAF_COUNT
) -> dict:
    """Orbax restore of the converted 839-leaf VAE checkpoint (flat ``vae/...`` tree)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint bytes absent at {path} "
            "(VAE_CHECKPOINT_BYTES_PRESENT=False; Stage D must provide/serve them)"
        )
    import orbax.checkpoint as ocp

    restored = None
    errors = []
    for loader in (
        lambda p: ocp.PyTreeCheckpointer().restore(str(p)),
        lambda p: ocp.StandardCheckpointer().restore(str(p)),
        lambda p: ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(str(p)),
    ):
        try:
            restored = loader(path)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if restored is None:
        raise RuntimeError("VAE restore failed with every durable handler: " + "; ".join(errors))
    state = _flatten_state(restored)
    if len(state) != expected_leaf_count:
        raise AssertionError(
            f"VAE restored leaf count {len(state)} != expected {expected_leaf_count}"
        )
    return state


def _np_dtype(dtype: str):
    if dtype == "bfloat16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    return dtype


def bind_vae_params(restored: dict, manifest: dict, dtype: str = VAE_DTYPE) -> dict:
    """Map the 728 runtime-required leaves into the runtime's ``p`` dict.

    ``restored`` is the flat 839-leaf tree; each manifest record's ``target`` key selects a
    converted leaf and ``source_key`` is the runtime ``p[...]`` key. The 111 converted leaves
    unused by the runtime (visual encoder paths) are intentionally not bound.
    """
    import numpy as np

    p: dict = {}
    missing = []
    for rec in manifest["records"]:
        target = rec["target"]
        if target not in restored:
            missing.append(target)
            continue
        p[rec["source_key"]] = np.asarray(restored[target]).astype(_np_dtype(dtype))
    if missing:
        raise AssertionError(
            f"VAE binding missing {len(missing)} target leaves, first={missing[:3]}"
        )
    if len(p) != VAE_RUNTIME_LEAF_COUNT:
        raise AssertionError(f"VAE bound param count {len(p)} != 728")
    return p


def build_vae_params_synthetic(manifest: dict, seed: int = 0, dtype: str = VAE_DTYPE) -> dict:
    """Synthetic correctly-shaped params for CPU interface tests only (never for Stage D)."""
    import numpy as np

    rng = np.random.RandomState(seed)
    p: dict = {}
    for rec in manifest["records"]:
        shape = tuple(int(s) for s in rec["expected_target_shape"])
        p[rec["source_key"]] = (rng.standard_normal(shape) * 0.02).astype(_np_dtype(dtype))
    return p


def restore_vae(
    *,
    vae_checkpoint: Optional[str] = None,
    manifest_path: Optional[str] = None,
    params: Optional[dict] = None,
    execute_compute: bool = False,
) -> VaeRuntime:
    """Resolve the VAE handle; restore+bind when computing with durable bytes."""
    rt = VaeRuntime(
        checkpoint_path=vae_checkpoint,
        manifest_path=manifest_path or DEFAULT_VAE_MANIFEST_HINT,
    )
    rt.assert_contract()
    rt.manifest = load_binding_manifest(rt.manifest_path)

    state = None
    if vae_checkpoint:
        cp = Path(vae_checkpoint)
        if execute_compute and not cp.exists():
            raise FileNotFoundError(
                "VAE checkpoint bytes absent (VAE_CHECKPOINT_BYTES_PRESENT=False); "
                "Stage D must provide/serve them before compute"
            )
        if cp.exists():
            state = restore_vae_checkpoint(cp)
    if execute_compute:
        if params is not None:
            rt.params = params
        elif state is not None:
            rt.params = bind_vae_params(state, rt.manifest)
        else:
            raise RuntimeError(
                "VAE execute_compute requires checkpoint bytes or explicit params; failing closed"
            )
        rt.decode_fn = load_vae_runtime().decode
        rt.bound = True
    return rt


def decode_latent(
    runtime: VaeRuntime,
    latent: Any,
    hooks: Optional[CaptureHooks] = None,
    *,
    execute_compute: bool = False,
) -> dict:
    """VAE decode with IDENTITY scaling; C6 fires only when a real tensor is produced."""
    runtime.assert_contract()
    if not execute_compute:
        return {
            "scaling_rule": runtime.scaling_rule,
            "dtype": runtime.dtype,
            "runtime_framework": runtime.runtime_framework,
            "compute": "DEFERRED",
            "bound": runtime.bound,
        }
    if runtime.decode_fn is None or runtime.params is None:
        raise RuntimeError("VAE runtime not bound for compute; failing closed")
    out = runtime.decode_fn(runtime.params, latent)
    if hooks is not None:
        hooks.c6_vae_output(out)
    return {"tensor": out, "scaling_rule": runtime.scaling_rule}


if __name__ == "__main__":
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np

    rt = restore_vae()
    res = decode_latent(rt, None, execute_compute=False)
    print("VAE_SCALING_RULE", rt.scaling_rule, "DEFERRED", res["compute"])
    print(
        "VAE_MANIFEST_COVERAGE",
        rt.manifest["runtime_binding_coverage"],
        "records",
        len(rt.manifest["records"]),
    )

    # CPU execution edge: synthetic params with REAL runtime shapes -> decode [1,128,2,2] ->
    # [1,3,32,32] (the exact structural case in VAE_SINGLE_TPU_RUNTIME_CLOSEOUT_AUTHORITY).
    params = build_vae_params_synthetic(rt.manifest, seed=3)
    rt2 = restore_vae(params=params, execute_compute=True)
    latent = np.zeros((1, 128, 2, 2), dtype="bfloat16")
    with __import__("tempfile").TemporaryDirectory() as td:
        from .capture_hooks import CaptureHooks

        hooks = CaptureHooks(td)
        out = decode_latent(rt2, latent, hooks=hooks, execute_compute=True)
        shape = tuple(int(s) for s in out["tensor"].shape)
        print("VAE_DECODE_EDGE", shape)
        assert shape == (1, 3, 32, 32), shape
        assert Path(td, "C6_vae_output.json").exists()
    print("VAE_DRIVER_STATIC=PASS")
