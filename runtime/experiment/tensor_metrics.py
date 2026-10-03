# -*- coding: utf-8 -*-
"""
Tensor record / A-B comparison utility for the G5 TPU blur causal experiment.

STANDALONE CPU UTILITY (numpy). Implements the Stage B tensor metrics contract
(runbook sections 9-10). The SAME implementation is used for both PATH A and PATH B.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _as_float64_finite_stats(a: np.ndarray) -> dict:
    f = a.astype(np.float64)
    return {
        "min": float(f.min()),
        "max": float(f.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
    }


def tensor_record(name: str, arr: np.ndarray) -> dict:
    """Metric record for one captured tensor.

    sha256 is computed over the canonical diagnostic payload: float32 little-endian
    contiguous bytes, prefixed with an ASCII header (name\0shape\tuple\0dtype\0).
    """
    a = np.asarray(arr)
    rec = {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "device": getattr(a, "device", "cpu"),
        "sharding": None,
        **_as_float64_finite_stats(a),
        "finite_count": int(np.isfinite(a).sum()),
        "nan_count": int(np.isnan(a).sum()),
        "posinf_count": int(np.isposinf(a).sum()),
        "neginf_count": int(np.isneginf(a).sum()),
    }
    payload = np.ascontiguousarray(a.astype(np.float32).ravel())
    h = hashlib.sha256()
    h.update(f"{name}\0{tuple(a.shape)}\0{str(a.dtype)}\0".encode("ascii"))
    h.update(payload.tobytes(order="C"))
    rec["sha256_diagnostic_payload"] = h.hexdigest()
    return rec


def compare_pair(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """A/B delta record (same implementation for both paths).

    exact comparison is meaningful only when shapes and dtypes match; NaN-safe
    different-element counting follows runbook contract.
    """
    fa = np.asarray(a, dtype=np.float64)
    fb = np.asarray(b, dtype=np.float64)
    if fa.shape != fb.shape:
        return {"name": name, "shape_mismatch": [list(fa.shape), list(fb.shape)]}
    delta = fb - fa
    exact_neq = a != b
    rec = {
        "name": name,
        "max_abs_delta": float(np.nanmax(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_delta": float(np.nanmean(np.abs(delta))) if delta.size else 0.0,
        "l1_delta": float(np.nansum(np.abs(delta))) if delta.size else 0.0,
        "l2_delta": float(np.sqrt(np.nansum(delta * delta))) if delta.size else 0.0,
        "different_element_count": int(np.count_nonzero(exact_neq)),
    }
    m = np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))
    if m and delta.size:
        a0, b0 = fa.ravel(), fb.ravel()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        rec["cosine_similarity"] = (
            float(np.dot(a0, b0) / denom) if denom > 0 else float("nan")
        )
    else:
        rec["cosine_similarity"] = None
    return rec


def compare_well_formed(record: dict) -> bool:
    required = {
        "name", "shape", "dtype", "min", "max", "mean", "std",
        "finite_count", "nan_count", "posinf_count", "neginf_count",
        "sha256_diagnostic_payload",
    }
    return required.issubset(record.keys())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = x.copy()
    y[0, 0] += 1e-3
    r = tensor_record("probe", x)
    print(r)
    print(compare_pair("probe", x, y))
    assert compare_well_formed(r)
    print("TENSOR_METRICS_VALIDATION=PASS")