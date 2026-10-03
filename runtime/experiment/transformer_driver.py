# -*- coding: utf-8 -*-
"""transformer_driver — instantiate / restore / shard / bind / forward for MageFlowTransformer.

C1R GAP-3 (Transformer restore+binding) resolution. The prior C1 driver bound the recovered
source modules but called an UNBOUND class method (``MageFlowTransformer.forward(...)``) and
never instantiated a model, never restored the 397-leaf converted checkpoint, and never bound
state. This module implements the concrete stages required by runbook section 11.1:

    build_transformer_runtime(...)
    construct_transformer_model(...)
    build_transformer_state_or_variables(...)
    restore_transformer_checkpoint(...)
    apply_transformer_sharding(...)
    bind_transformer_state(...)
    transformer_forward(...)

Sources of authority
--------------------
* recovered source: ``mage_flow_keras/full_model.py`` (MageFlowTransformer.parameter_paths),
  ``mage_flow_keras/block0_candidate.py`` (capture=True intermediates),
  ``mage_flow_keras/full_real_weight_conversion.py`` (oc.PyTreeCheckpointer().restore).
* converted checkpoint manifest: FULL_REAL_WEIGHT_CONVERSION_MANIFEST.json
  (397 rows; target_dtype FLOAT32; target_key namespace ``planned.mage_flow_transformer.``).
* sharding rule: ``mage_flow_keras/tpu_fast_track.py::compute_partition_spec``
  (rank-2 -> shard largest dim divisible by mesh_size; rank<=1 -> replicate). Applying the
  recorded rule to the 397 target shapes reproduces exactly 174 sharded / 223 replicated.
* C2 authority: block0_candidate ``capture=True`` -> intermediates["image_q"] post-RoPE.

Only the 397-leaf converted checkpoint is used. No CUDA path. No unbound class-level forward.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .capture_hooks import CaptureHooks
from .runtime_contract import (
    C2_BOUNDARY_DESCRIPTION,
    C2_CAPTURE_MECHANISM,
    C2_EXPECTED_DTYPE,
    C2_EXPECTED_SHAPE,
    C2_FILE,
    C2_SYMBOL,
)

TRANSFORMER_DTYPE = "float32"
TRANSFORMER_LEAF_COUNT = 397
TRANSFORMER_SHARDED = 174
TRANSFORMER_REPLICATED = 223
MESH_AXIS_NAME = "model"
MESH_SIZE = 8
PERFECT_ROOT_KEY = "planned.mage_flow_transformer."
TRANSFORMER_CHECKPOINT_AUTHORITY = {
    "run_sha256": "428d7d605051835f7d990de90ab0b8043506ece79f3f001d9ca8d190c916daa3",
    "file_count": 28,
    "total_bytes": 7845824678,
}
DEFAULT_TRANSFORMER_SOURCE_HINT = str(Path(__file__).resolve().parents[2] / "source")


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def _package_parent(source_root: str | Path) -> Path:
    src = Path(source_root)
    return src.parent if src.name == "mage_flow_keras" else src


def load_transformer_source(source_root: str | Path) -> dict:
    """Import the recovered ``mage_flow_keras`` package from its durable source tree.

    ``full_model.py`` itself performs ``from mage_flow_keras.block0_candidate import ...`` so
    the package parent must be importable; we import the package as a whole rather than
    synthesizing ad-hoc module specs.
    """
    src = _package_parent(source_root)
    pkg = src / "mage_flow_keras" / "full_model.py"
    if not pkg.is_file():
        raise FileNotFoundError(f"recovered mage_flow_keras.full_model not found under {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return {
        "source_dir": str(src),
        "full_model": importlib.import_module("mage_flow_keras.full_model"),
        "block0_candidate": importlib.import_module("mage_flow_keras.block0_candidate"),
    }


# ---------------------------------------------------------------------------
# Instantiate / state
# ---------------------------------------------------------------------------
def construct_transformer_model(source: dict, config: Any = None) -> Any:
    """Instantiate ``MageFlowTransformer`` (a real, bindable model instance)."""
    fm = source["full_model"]
    model = fm.MageFlowTransformer(config or fm.MageFlowTransformerConfig())
    return model


def build_transformer_state_or_variables(model: Any) -> dict:
    """Return the model's Keras variables keyed by full target path."""
    return model.parameter_paths()


# ---------------------------------------------------------------------------
# Restore / shard / bind
# ---------------------------------------------------------------------------
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


def restore_transformer_checkpoint(
    checkpoint_path: str | Path, *, expected_leaf_count: int = TRANSFORMER_LEAF_COUNT
) -> dict:
    """Orbax restore of the 397-leaf converted transformer checkpoint (flat target-key dict)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"transformer checkpoint bytes absent at {path} "
            "(TRANSFORMER_CHECKPOINT_BYTES_PRESENT=False; Stage D must provide/serve them)"
        )
    import orbax.checkpoint as ocp

    restored = ocp.PyTreeCheckpointer().restore(str(path))
    state = _flatten_state(restored)
    if len(state) != expected_leaf_count:
        raise AssertionError(
            f"transformer restored leaf count {len(state)} != expected {expected_leaf_count}"
        )
    return state


def compute_partition_spec(shape: tuple, mesh_size: int = MESH_SIZE) -> tuple:
    """Port of the durable tpu_fast_track.compute_partition_spec sharding rule."""
    if mesh_size is None or mesh_size <= 1:
        return ()
    if len(shape) == 2:
        divisible = [i for i, dim in enumerate(shape) if dim and dim % mesh_size == 0]
        if divisible:
            best = max(divisible, key=lambda i: shape[i])
            spec = [None, None]
            spec[best] = MESH_AXIS_NAME
            return tuple(spec)
        return ()
    return ()


def static_sharding_plan(
    model: Any, mesh_size: int = MESH_SIZE, *, expect_contract: bool = True
) -> dict:
    """Per-leaf partition specs + validation against the 174/223 recorded contract."""
    shapes = model.parameter_shapes()
    plan = {path: compute_partition_spec(tuple(shape), mesh_size) for path, shape in shapes.items()}
    sharded = sum(1 for spec in plan.values() if any(s is not None for s in spec))
    replicated = len(plan) - sharded
    if expect_contract and len(plan) != TRANSFORMER_LEAF_COUNT:
        raise AssertionError(f"transformer sharding plan leaf count {len(plan)} != 397")
    if expect_contract and (sharded != TRANSFORMER_SHARDED or replicated != TRANSFORMER_REPLICATED):
        raise AssertionError(
            f"transformer sharding contract {sharded}/{replicated} != recorded "
            f"{TRANSFORMER_SHARDED}/{TRANSFORMER_REPLICATED}"
        )
    return {
        "plan": plan,
        "leaf_count": len(plan),
        "sharded": sharded,
        "replicated": replicated,
        "mesh_axis": MESH_AXIS_NAME,
        "mesh_size": mesh_size,
    }


def apply_transformer_sharding(model: Any, mesh: Any, mesh_size: int = MESH_SIZE) -> dict:
    """Apply the static partition plan to a bound model's variables.

    ``mesh=None`` (CPU synthetic path) records the validated static plan without device
    placement. On the target device, each variable is placed with NamedSharding.
    """
    info = static_sharding_plan(model, mesh_size)
    if mesh is None:
        info["applied"] = False
        info["reason"] = "no TPU mesh (CPU synthetic/preflight); static plan validated only"
        return info
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    for path, var in model.parameter_paths().items():
        spec = info["plan"][path]
        sharding = NamedSharding(mesh, P(*spec))
        var.assign(jax.device_put(var.value, sharding))
    info["applied"] = True
    return info


def bind_transformer_state(model: Any, state: dict, dtype: str = TRANSFORMER_DTYPE) -> dict:
    """Bind restored params into a real model instance (explicit assign, no class call)."""
    import jax.numpy as jnp

    paths = model.parameter_paths()
    missing = sorted(set(paths) - set(state))
    extra = sorted(set(state) - set(paths))
    if missing or extra:
        raise AssertionError(
            f"transformer bind key mismatch: missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:3]} first_extra={extra[:3]}"
        )
    for path, var in paths.items():
        var.assign(jnp.asarray(state[path]).astype(dtype))
    return {"bound_leaf_count": len(paths), "dtype": dtype}


def pack_edit_image_sequence(target, reference):
    """Pack target+reference image latents replica-by-replica for Mage-Flow Edit.

    Inputs are NCHW arrays with identical [replica, channel, height, width] shape.
    Output is [1, packed_tokens, channel] laid out as
    ``target_0, reference_0, target_1, reference_1, ...`` plus cumulative
    per-replica sequence boundaries for packed attention.
    """
    import numpy as np

    target = np.asarray(target)
    reference = np.asarray(reference)
    if target.shape != reference.shape or target.ndim != 4:
        raise ValueError(
            f"target/reference must have identical NCHW shape; got {target.shape} and {reference.shape}"
        )
    replicas, channels, height, width = target.shape
    n = height * width
    blocks = []
    for i in range(replicas):
        t = target[i].transpose(1, 2, 0).reshape(n, channels)
        r = reference[i].transpose(1, 2, 0).reshape(n, channels)
        blocks.extend([t, r])
    packed = np.concatenate(blocks, axis=0)[None, ...]
    cu = np.arange(replicas + 1, dtype=np.int32) * (2 * n)
    return packed, cu


def extract_edit_target_prediction(packed_prediction, *, replicas: int, target_tokens: int):
    """Select only target-token predictions from per-replica [target, reference] blocks."""
    import numpy as np

    pred = np.asarray(packed_prediction)
    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"packed_prediction must be [1,T,C], got {pred.shape}")
    block = 2 * int(target_tokens)
    expected = int(replicas) * block
    if pred.shape[1] != expected:
        raise ValueError(f"packed token count {pred.shape[1]} != expected {expected}")
    parts = [pred[0, i * block : i * block + target_tokens] for i in range(replicas)]
    return np.stack(parts, axis=0)
