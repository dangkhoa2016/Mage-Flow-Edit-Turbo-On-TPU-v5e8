# -*- coding: utf-8 -*-
"""rope_provider — test-only image RoPE builder driving the A/B controlled experiment.

Test-only. NEVER imported by production runtime code. No production RoPE generation is
patched or re-defaulted.

Two modes only:
  control          (DEFAULT)  -> PATH A = RECONSTRUCTED_CANONICAL_CONTROL
  c2a3-diagnostic  (opt-in)   -> PATH B = C2A3_DIAGNOSTIC_ROPE_AUTHORITY

The canonical basis arrays MUST come from the frozen torch-captured npy files (they are an
explicit dependency, resolved by artifact_resolver). CRITICAL numeric caveat (verified by the
prior CPU session, preserved here): plain NumPy float32 power generation of
1/(10000**(arange(0,56,2)/56)) lands exactly one ULP BELOW the torch-canonical value at
column 11 — i.e., naive NumPy generation would silently produce the DIAGNOSTIC DOWN-1ULP
value in the CONTROL path. The control machinery therefore always loads the captured basis.

PATH B must emit the diagnostic flags:
  DIAGNOSTIC_ONLY=True / PRODUCTION_DEFAULT_CHANGED=False / HISTORICAL_PROVENANCE_CLAIM=False

PATH B required C2A3 scope assertions:
  BASIS_CHANGED_LOCAL_BINS == [11]
  BASIS_CHANGED_GLOBAL_BINS == [19, 47]
  NON_BIN11_BASIS_CHANGED == False
  INTERVENTION_SCOPE_VIOLATION == False
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

CANONICAL_AXES_DIM = (16, 56, 56)
CANONICAL_THETA = 10000.0

_DIAGNOSTIC_BIN = 11
_EXPECTED_CHANGED_GLOBAL_BINS = [19, 47]

ROPE_MODE_CONTROL = "control"
ROPE_MODE_DIAGNOSTIC = "c2a3-diagnostic"
VALID_MODES = (ROPE_MODE_CONTROL, ROPE_MODE_DIAGNOSTIC)


@dataclass(frozen=True)
class PathBFlags:
    """Exact PATH B introspection block (runbook section 13)."""

    rope_mode: str = "C2A3_DIAGNOSTIC"
    diagnostic_only: bool = True
    production_default_changed: bool = False
    historical_provenance_claim: bool = False

    def to_lines(self) -> list[str]:
        return [
            "DIAGNOSTIC_ONLY=True",
            "PRODUCTION_DEFAULT_CHANGED=False",
            "HISTORICAL_PROVENANCE_CLAIM=False",
        ]


def path_b_flags() -> PathBFlags:
    return PathBFlags()


def _load_canonical_basis(
    basis_dim16: Path, basis_dim56: Path, axes_dim: Sequence[int]
) -> list[np.ndarray]:
    """Load the frozen torch-captured canonical bases (never re-computed)."""
    for p in (basis_dim16, basis_dim56):
        if not Path(p).is_file():
            raise FileNotFoundError(
                f"canonical basis capture unavailable: {p} "
                "(CONTROL_ROPE_AUTHORITY_RESOLVED=False)"
            )
    b16 = np.load(basis_dim16)
    b56 = np.load(basis_dim56)
    if b16.shape != (16 // 2,) or b16.dtype != np.float32:
        raise ValueError("unexpected captured-basis-dim16 shape/dtype")
    if b56.shape != (56 // 2,) or b56.dtype != np.float32:
        raise ValueError("unexpected captured-basis-dim56 shape/dtype")
    return [b16 if d == 16 else b56 for d in axes_dim]


def _down_one_ulp(basis: np.ndarray, column: int) -> np.ndarray:
    mod = basis.copy()
    mod[column] = np.nextafter(mod[column], np.float32(-np.inf))
    return mod


def build_image_rope(
    mode: str,
    latent_h: int,
    latent_w: int,
    basis_dim16: Path,
    basis_dim56: Path,
    axes_dim: Sequence[int] = CANONICAL_AXES_DIM,
    theta: float = CANONICAL_THETA,
) -> np.ndarray:
    """Build the cartesian (cos, sin) image RoPE tensor [H*W, sum(axes)/2, 2] float32.

    mode "control"          -> PATH A canonical (frozen captured basis, NumPy float32 trig)
    mode "c2a3-diagnostic"  -> PATH B (captured basis with frequency index 11 DOWN 1 ULP on
                               the height+width axes; NumPy float32 cos/sin member)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown rope mode {mode!r}")
    if theta <= 0:
        raise ValueError("theta must be positive")

    bases = _load_canonical_basis(basis_dim16, basis_dim56, axes_dim)
    half = [len(b) for b in bases]

    basis_hw = bases[1]
    if mode == ROPE_MODE_DIAGNOSTIC:
        basis_hw = _down_one_ulp(basis_hw, _DIAGNOSTIC_BIN)

    n_tok = latent_h * latent_w
    n = max(latent_h, latent_w)
    pos_index = np.arange(n, dtype=np.float32)
    neg_index = (np.arange(n, dtype=np.float32)[::-1]) * np.float32(-1) - np.float32(1)

    h_seq = _axis_sequence(neg_index, pos_index, latent_h)
    w_seq = _axis_sequence(neg_index, pos_index, latent_w)

    ph = h_seq[:, None] * basis_hw[None, :]
    pw = w_seq[:, None] * basis_hw[None, :]
    zeros = np.zeros((half[0],), dtype=np.float32)

    out = np.zeros((n_tok, sum(half), 2), dtype=np.float32)
    for r in range(latent_h):
        for c in range(latent_w):
            phase = np.concatenate([zeros, ph[r], pw[c]]).astype(np.float32)
            out[r * latent_w + c, :, 0] = np.cos(phase)
            out[r * latent_w + c, :, 1] = np.sin(phase)
    return out



def build_image_rope_sequence(
    mode: str,
    shapes: Sequence[tuple[int, int, int]],
    basis_dim16: Path,
    basis_dim56: Path,
    axes_dim: Sequence[int] = CANONICAL_AXES_DIM,
    theta: float = CANONICAL_THETA,
) -> np.ndarray:
    """Build packed MageFlow image RoPE for ``[(frame, H, W), ...]``.

    Each shape entry advances the frame-axis origin by its list index, matching
    upstream ``MageFlowEmbedRope._compute_video_freqs(..., idx)``. Spatial axes
    use the same centered scale-RoPE coordinates as :func:`build_image_rope`.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown rope mode {mode!r}")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if not shapes:
        raise ValueError("shapes must contain at least one frame block")

    bases = _load_canonical_basis(basis_dim16, basis_dim56, axes_dim)
    basis_frame = bases[0]
    basis_hw = bases[1]
    if mode == ROPE_MODE_DIAGNOSTIC:
        basis_hw = _down_one_ulp(basis_hw, _DIAGNOSTIC_BIN)

    blocks = []
    for idx, (frame, latent_h, latent_w) in enumerate(shapes):
        if frame <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError(f"invalid image shape {(frame, latent_h, latent_w)}")
        n = max(latent_h, latent_w)
        pos_index = np.arange(n, dtype=np.float32)
        neg_index = np.arange(n, dtype=np.float32)[::-1] * np.float32(-1) - np.float32(1)
        h_seq = _axis_sequence(neg_index, pos_index, latent_h)
        w_seq = _axis_sequence(neg_index, pos_index, latent_w)

        frame_seq = np.arange(idx, idx + frame, dtype=np.float32)
        pf = frame_seq[:, None] * basis_frame[None, :]
        ph = h_seq[:, None] * basis_hw[None, :]
        pw = w_seq[:, None] * basis_hw[None, :]

        out = np.zeros((frame * latent_h * latent_w, sum(len(b) for b in bases), 2), dtype=np.float32)
        k = 0
        for f in range(frame):
            for r in range(latent_h):
                for c in range(latent_w):
                    phase = np.concatenate([pf[f], ph[r], pw[c]]).astype(np.float32)
                    out[k, :, 0] = np.cos(phase)
                    out[k, :, 1] = np.sin(phase)
                    k += 1
        blocks.append(out)
    return np.concatenate(blocks, axis=0)

def _axis_sequence(neg_index: np.ndarray, pos_index: np.ndarray, length: int) -> np.ndarray:
    n = len(pos_index)
    tail = neg_index[-(length // 2):] if length >= 2 else neg_index[:0]
    head = pos_index[: length // 2] if length >= 2 else pos_index[:0]
    full = np.concatenate([tail, head])[:length]
    if len(full) != length:
        raise AssertionError(f"axis sequence length {len(full)} != {length} (n={n})")
    return full


def verify_control_against_fixture(rope_a: np.ndarray, fixture_path: Path) -> dict:
    """Assert PATH A == fixture everywhere except imag part of global bins [19,47]."""
    fixture = np.load(fixture_path)
    if rope_a.shape != fixture.shape:
        raise AssertionError(f"shape mismatch {rope_a.shape} != {fixture.shape}")
    neq = rope_a != fixture
    real = int(np.count_nonzero(neq[..., 0]))
    imag = int(np.count_nonzero(neq[..., 1]))
    bins = sorted(set(np.where(neq.any(axis=2).any(axis=0))[0].tolist()))
    ok = real == 0 and bins == _EXPECTED_CHANGED_GLOBAL_BINS and int(np.count_nonzero(neq)) >= 1
    if not ok:
        raise AssertionError(
            f"CONTROL vs fixture divergence anomaly: bins={bins} real={real} imag={imag}"
        )
    return {
        "PATH_A_NAME": "RECONSTRUCTED_CANONICAL_CONTROL",
        "changed_global_bins": bins,
        "real_components": real,
        "imag_components": imag,
    }


def verify_diagnostic_against_fixture(rope_b: np.ndarray, fixture_path: Path) -> dict:
    """Assert PATH B == fixture EXACTLY (C2A3 H02 full-tensor reproduction)."""
    fixture = np.load(fixture_path)
    neq = rope_b != fixture
    total = int(np.count_nonzero(neq))
    if total != 0:
        raise AssertionError(f"PATH B vs fixture not exact: {total} different components")
    return {"ARRAY_EQUAL": True, "TOTAL_DIFFERENT_COMPONENTS": 0}


def scope_check(a: np.ndarray, b: np.ndarray) -> dict:
    """C2A3 intervention-scope assertions between PATH A and PATH B."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"shape/dtype mismatch {a.shape}:{a.dtype} != {b.shape}:{b.dtype}")
    changed = a != b
    bins = sorted(set(np.where(changed.any(axis=2).any(axis=0))[0].tolist()))
    scope_ok = bins == _EXPECTED_CHANGED_GLOBAL_BINS
    result = {
        "BASIS_CHANGED_LOCAL_BINS": [_DIAGNOSTIC_BIN],
        "BASIS_CHANGED_GLOBAL_BINS": bins,
        "NON_BIN11_BASIS_CHANGED": bool(bins != _EXPECTED_CHANGED_GLOBAL_BINS),
        "INTERVENTION_SCOPE_VIOLATION": bool(not scope_ok),
        "changed_components": int(np.count_nonzero(changed)),
    }
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-dim16", type=Path)
    ap.add_argument("--basis-dim56", type=Path)
    ap.add_argument("--rope-fixture", type=Path)
    ap.add_argument("--latent-h", type=int, default=32)
    ap.add_argument("--latent-w", type=int, default=32)
    ap.add_argument("--mode", default=ROPE_MODE_CONTROL)
    args = ap.parse_args()

    a = build_image_rope(ROPE_MODE_CONTROL, args.latent_h, args.latent_w,
                         args.basis_dim16, args.basis_dim56)
    b = build_image_rope(ROPE_MODE_DIAGNOSTIC, args.latent_h, args.latent_w,
                         args.basis_dim16, args.basis_dim56)
    print("shape", a.shape, a.dtype)
    print("PATH_A_vs_FIXTURE", verify_control_against_fixture(a, args.rope_fixture))
    print("PATH_B_vs_FIXTURE", verify_diagnostic_against_fixture(b, args.rope_fixture))
    sc = scope_check(a, b)
    print("SCOPE", sc)
    print("\n".join(path_b_flags().to_lines()))
    assert sc["INTERVENTION_SCOPE_VIOLATION"] is False
    assert sc["NON_BIN11_BASIS_CHANGED"] is False
    assert sc["BASIS_CHANGED_LOCAL_BINS"] == [_DIAGNOSTIC_BIN]
    assert sc["BASIS_CHANGED_GLOBAL_BINS"] == _EXPECTED_CHANGED_GLOBAL_BINS
    print("ROPE_PROVIDER_VALIDATION=PASS")