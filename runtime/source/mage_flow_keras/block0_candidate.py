"""Mage-Flow Block-0 implementation in Keras 3 / JAX.

The implementation reproduces the pinned upstream ``MageFlowTransformerBlock``
(math) for the smallest real implementation unit required by Level-2 parity:
a single dual-stream transformer block (block index 0) with

  * image/text adaptive modulation (SiLU + ``Linear(dim, 6*dim)``),
  * affine-free LayerNorm (eps=1e-6) with shift/scale/gate modulation,
  * joint double-stream attention (image ``to_q/to_k/to_v``, text
    ``add_q_proj/add_k_proj/add_v_proj``),
  * QK RMSNorm for both streams,
  * image-only rotary embedding (adjacent-pair complex convention, real math),
  * joint packed text-then-image attention with FA2-equivalent semantics
    (scaled dot-product, non-causal, dropout=0, default
    ``1/sqrt(head_dim)`` scale, per-sample segment isolation),
  * output projections (``to_out.0`` image, ``to_add_out`` text),
  * gated attention residual then gated MLP residual for both streams,
  * output ordering ``(text_stream, image_stream)``.

Design rules:
  * Implemented in the project's Keras 3 / JAX stack.  ``jax.numpy`` is used
    for the low-level array math because Keras' ``scatter_update`` wrapper does
    not preserve flat first-axis index sets; the module still runs on the JAX
    backend and is exposed as a ``keras.Layer``.
  * ``call(...)`` is the Keras entrypoint and maps onto ``forward(...)``.
  * Intermediate tensors are exposed through ``capture=True`` (off by default,
    deterministic, not used by the production path).
  * Input validation fails closed with actionable messages.
"""

from __future__ import annotations

import math
from typing import Any

import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import keras

# Source-fidelity invariants (asserted by tests).
BLOCK0_OUTPUT_ORDER = ("text", "image")
JOINT_PACK_ORDER = ("text", "image")
ATTENTION_CAUSAL = False
ATTENTION_DROPOUT = 0.0
SOFTMAX_SCALE_MODE = "1/sqrt(head_dim)"

# Block parameter leaf table (single source of truth for the 32 per-block
# Keras parameters, shared by the full-model spec generator).
_PREFIX_BLOCK = "blocks."


def block_parameter_leaf_shapes(
    dim: int, head_dim: int
) -> dict[str, tuple[int, ...]]:
    """Return the 32 ordered ``{leaf_path: shape}`` entries of one transformer
    block (Keras ``[in, out]`` convention), identical for every block index."""
    inner = dim
    mlp_inner = 4 * dim
    hd = int(head_dim)
    leaves: dict[str, tuple[int, ...]] = {
        "image_modulation.kernel": (dim, 6 * dim),
        "image_modulation.bias": (6 * dim,),
        "text_modulation.kernel": (dim, 6 * dim),
        "text_modulation.bias": (6 * dim,),
        "image_mlp.gate_projection.kernel": (dim, mlp_inner),
        "image_mlp.gate_projection.bias": (mlp_inner,),
        "image_mlp.output_projection.kernel": (mlp_inner, dim),
        "image_mlp.output_projection.bias": (dim,),
        "text_mlp.gate_projection.kernel": (dim, mlp_inner),
        "text_mlp.gate_projection.bias": (mlp_inner,),
        "text_mlp.output_projection.kernel": (mlp_inner, dim),
        "text_mlp.output_projection.bias": (dim,),
        "attention.image_q.kernel": (dim, inner),
        "attention.image_q.bias": (inner,),
        "attention.image_k.kernel": (dim, inner),
        "attention.image_k.bias": (inner,),
        "attention.image_v.kernel": (dim, inner),
        "attention.image_v.bias": (inner,),
        "attention.image_out.kernel": (inner, dim),
        "attention.image_out.bias": (dim,),
        "attention.text_q.kernel": (dim, inner),
        "attention.text_q.bias": (inner,),
        "attention.text_k.kernel": (dim, inner),
        "attention.text_k.bias": (inner,),
        "attention.text_v.kernel": (dim, inner),
        "attention.text_v.bias": (inner,),
        "attention.text_out.kernel": (inner, dim),
        "attention.text_out.bias": (dim,),
        "attention.image_q_norm.scale": (hd,),
        "attention.image_k_norm.scale": (hd,),
        "attention.text_q_norm.scale": (hd,),
        "attention.text_k_norm.scale": (hd,),
    }
    if len(leaves) != 32:
        raise MageFlowBlock0Error(
            f"block parameter leaf table must contain exactly 32 entries, "
            f"got {len(leaves)}"
        )
    return leaves


class MageFlowBlock0Error(RuntimeError):
    """Raised when Block-0 candidate inputs or configuration are invalid."""


def _as_i32(x: Any, name: str) -> jnp.ndarray:
    arr = jnp.asarray(x)
    if arr.ndim != 1:
        raise MageFlowBlock0Error(
            f"{name}: expected a 1-D cumulative-length array, got rank {arr.ndim}"
        )
    if arr.dtype.kind not in ("i", "u"):
        raise MageFlowBlock0Error(
            f"{name}: cu_seqlens dtype must be integer (int32/int64), got {arr.dtype}"
        )
    return arr.astype(jnp.int32)


def _validate_cu_lens(
    cu: Any,
    name: str,
    num_tokens: int,
    batch: int,
) -> jnp.ndarray:
    arr = _as_i32(cu, name)
    n = arr.shape[0]
    if n != batch + 1:
        raise MageFlowBlock0Error(
            f"{name}: expected batch+1 entries ({batch + 1}), got {n}"
        )
    if int(arr[0]) != 0:
        raise MageFlowBlock0Error(f"{name}: first entry must be 0, got {int(arr[0])}")
    if int(arr[-1]) != num_tokens:
        raise MageFlowBlock0Error(
            f"{name}: terminal cumulative count {int(arr[-1])} does not match "
            f"stream token count {num_tokens}"
        )
    deltas = arr[1:] - arr[:-1]
    if bool(jnp.any(deltas <= 0)):
        raise MageFlowBlock0Error(f"{name}: cu_seqlens must be strictly increasing")
    return arr


def _layernorm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y.astype(x.dtype)


def _rms_norm(x: jnp.ndarray, eps: float, scale: jnp.ndarray | None = None) -> jnp.ndarray:
    x = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    y = x * jax.lax.rsqrt(var + eps)
    if scale is not None:
        y = y * scale.astype(jnp.float32)
    return y.astype(x.dtype)


def _gelu_tanh(x: jnp.ndarray) -> jnp.ndarray:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + jnp.tanh(c * (x + 0.044715 * jnp.power(x, 3))))


def _apply_rotary(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Apply image-only rotary embedding (adjacent-pair complex convention).

    Matches ``apply_rotary_emb_mageflow``: ``view_as_complex`` (last-dim
    adjacent pairs) multiplied by ``freqs`` then ``view_as_real`` flattened.
    ``freqs`` is the cartesian (cos, sin) real representation with shape
    ``[N_tokens, head_dim // 2, 2]``.
    """
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    pair = jnp.reshape(x, x.shape[:-1] + (half, 2))
    re, im = pair[..., 0], pair[..., 1]
    cos = freqs[..., 0][:, None, ...]
    sin = freqs[..., 1][:, None, ...]
    out_re = re * cos - im * sin
    out_im = re * sin + im * cos
    return jnp.stack([out_re, out_im], axis=-1).reshape(x.shape)



class MageFlowBlock0Candidate(keras.Layer):
    """Implementation body added in the next history step."""
    pass
