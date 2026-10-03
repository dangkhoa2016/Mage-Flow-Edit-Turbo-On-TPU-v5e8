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
    """Keras 3 / JAX candidate for ``MageFlowTransformerBlock`` (block 0).

    The constructor mirrors the pinned upstream constructor::

        MageFlowTransformerBlock(dim, num_attention_heads, attention_head_dim, eps=1e-6)

    ``call`` is the Keras entrypoint and maps 1:1 onto ``forward``.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        block_index: int = 0,
        name: str = "mage_flow_block0_candidate",
    ):
        super().__init__(name=name, dtype="float32")
        if keras.config.backend() != "jax":
            raise MageFlowBlock0Error(
                "MageFlowBlock0Candidate requires the JAX backend "
                f"(KERAS_BACKEND=jax); active backend is {keras.config.backend()}"
            )
        self.dim = int(dim)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.eps = float(eps)
        self.block_index = int(block_index)
        if self.block_index < 0:
            raise MageFlowBlock0Error(
                f"block_index must be >= 0, got {self.block_index}"
            )
        self._block_prefix = f"{_PREFIX_BLOCK}{self.block_index}."
        inner = self.num_attention_heads * self.attention_head_dim
        if inner != self.dim:
            raise MageFlowBlock0Error(
                f"dim {self.dim} must equal heads*head_dim "
                f"({self.num_attention_heads}*{self.attention_head_dim}={inner})"
            )
        if self.attention_head_dim % 2 != 0:
            raise MageFlowBlock0Error(
                "attention_head_dim must be even for the adjacent-pair rotary convention"
            )
        # Source-fidelity switches (asserted by tests).
        self._causal = ATTENTION_CAUSAL
        self._dropout = ATTENTION_DROPOUT
        self._softmax_scale = self.attention_head_dim ** -0.5
        self._param_paths: dict[str, keras.Variable] = {}
        self._build_parameters()

    # ------------------------------------------------------------------
    # Parameter construction (Keras convention kernels [in, out]).
    # ------------------------------------------------------------------
    def _add_param(self, dotted_path: str, shape: tuple[int, ...]) -> keras.Variable:
        var = self.add_weight(
            name=dotted_path,
            shape=shape,
            initializer="zeros",
            trainable=True,
        )
        self._param_paths[dotted_path] = var
        return var

    def _build_parameters(self) -> None:
        for leaf, shape in block_parameter_leaf_shapes(
            self.dim, self.attention_head_dim
        ).items():
            self._add_param(f"{self._block_prefix}{leaf}", shape)

    # ------------------------------------------------------------------
    # Introspection helpers used by the weight loader.
    # ------------------------------------------------------------------
    @property
    def softmax_scale(self) -> float:
        return self._softmax_scale

    @property
    def causal(self) -> bool:
        return self._causal

    @property
    def dropout_rate(self) -> float:
        return self._dropout

    def parameter_paths(self) -> dict[str, keras.Variable]:
        """Return {dotted candidate path: Keras variable}.

        Dotted paths match the mapping-file target leaf used by
        ``reference/transformer_mapping.json`` (``planned.`` prefix removed).
        """
        return dict(self._param_paths)

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {path: tuple(var.shape) for path, var in self._param_paths.items()}

    # ------------------------------------------------------------------
    # Dense helpers (Keras convention: x @ kernel + bias).
    # ------------------------------------------------------------------
    def _dense(self, x: jnp.ndarray, path_prefix: str) -> jnp.ndarray:
        kernel = jnp.asarray(self._param_paths[f"{path_prefix}.kernel"])
        bias = jnp.asarray(self._param_paths[f"{path_prefix}.bias"])
        return jnp.matmul(x, kernel) + bias

    # ------------------------------------------------------------------
    # Joint packed attention primitives.
    # ------------------------------------------------------------------
    def _joint_pack_indices(
        self,
        txt_cu: jnp.ndarray,
        img_cu: jnp.ndarray,
        n_txt: int,
        n_img: int,
        batch: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (txt_dest, img_dest, joint_cu, joint_lens) exact to upstream.

        Per sample the joint order is ``[text tokens..., image tokens...]``.
        ``txt_lens`` / ``img_lens`` are ``cu[1:] - cu[:-1]``.
        """
        txt_lens = txt_cu[1:] - txt_cu[:-1]
        img_lens = img_cu[1:] - img_cu[:-1]
        joint_lens = txt_lens + img_lens
        joint_cu = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(joint_lens, dtype=jnp.int32)],
            axis=0,
        )
        sample_indices = jnp.arange(batch, dtype=jnp.int32)
        txt_sample_ids = jnp.repeat(sample_indices, txt_lens)
        img_sample_ids = jnp.repeat(sample_indices, img_lens)
        txt_intra = jnp.arange(n_txt, dtype=jnp.int32) - txt_cu[txt_sample_ids]
        img_intra = jnp.arange(n_img, dtype=jnp.int32) - img_cu[img_sample_ids]
        txt_dest = joint_cu[txt_sample_ids] + txt_intra
        img_dest = joint_cu[img_sample_ids] + txt_lens[img_sample_ids] + img_intra
        return txt_dest, img_dest, joint_cu, joint_lens

    def _segment_attention(
        self,
        joint_query: jnp.ndarray,
        joint_key: jnp.ndarray,
        joint_value: jnp.ndarray,
        joint_cu: jnp.ndarray,
        batch: int,
        *,
        attention_mode: str = "masked-global",
        query_chunk: int = 256,
    ) -> jnp.ndarray:
        """FA2-equivalent packed attention (scaled SDPA, non-causal, dropout 0).

        Equivalent to ``flash_attn_varlen_func`` with ``causal=False``,
        ``dropout_p=0.0`` and default ``softmax_scale=1/sqrt(head_dim)``.
        Cross-sample attention is fully masked (per-segment softmax), so there
        is no cross-sample leakage.
        """
        mode = str(attention_mode).strip().lower()
        if mode == "masked-global":
            lens = joint_cu[1:] - joint_cu[:-1]
            sample_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), lens)
            logits = (
                jnp.einsum("ihd,jhd->ihj", joint_query, joint_key)
                * self._softmax_scale
            )
            mask = (sample_ids[:, None] == sample_ids[None, :])[:, None, :]
            logits = jnp.where(
                mask, logits, jnp.array(-jnp.inf, dtype=logits.dtype)
            )
            attn = jax.nn.softmax(logits, axis=-1)
            return jnp.einsum("ihj,jhd->ihd", attn, joint_value)

        if mode not in {"segmented", "segmented-query-chunk"}:
            raise MageFlowBlock0Error(
                "attention_mode must be one of "
                "{'masked-global','segmented','segmented-query-chunk'}, "
                f"got {attention_mode!r}"
            )
        if batch <= 0:
            raise MageFlowBlock0Error(f"batch must be > 0, got {batch}")

        total = int(joint_query.shape[0])
        if total % batch != 0:
            raise MageFlowBlock0Error(
                f"{mode}: total joint tokens {total} not divisible by batch {batch}"
            )
        segment_len = total // batch
        lens = joint_cu[1:] - joint_cu[:-1]
        if not bool(jnp.all(lens == segment_len)):
            raise MageFlowBlock0Error(
                f"{mode}: equal packed request lengths are required; "
                f"expected every joint segment to contain {segment_len} tokens"
            )

        chunk = int(query_chunk)
        if mode == "segmented-query-chunk" and chunk <= 0:
            raise MageFlowBlock0Error(
                f"query_chunk must be > 0 for segmented-query-chunk, got {chunk}"
            )

        outputs = []
        for segment_index in range(batch):
            lo = segment_index * segment_len
            hi = lo + segment_len
            key = joint_key[lo:hi]
            value = joint_value[lo:hi]

            if mode == "segmented":
                query = joint_query[lo:hi]
                logits = (
                    jnp.einsum("ihd,jhd->ihj", query, key)
                    * self._softmax_scale
                )
                attn = jax.nn.softmax(logits, axis=-1)
                outputs.append(jnp.einsum("ihj,jhd->ihd", attn, value))
                continue

            query_outputs = []
            for query_lo in range(lo, hi, chunk):
                query_hi = min(query_lo + chunk, hi)
                query = joint_query[query_lo:query_hi]
                logits = (
                    jnp.einsum("ihd,jhd->ihj", query, key)
                    * self._softmax_scale
                )
                attn = jax.nn.softmax(logits, axis=-1)
                query_outputs.append(
                    jnp.einsum("ihj,jhd->ihd", attn, value)
                )
            outputs.append(jnp.concatenate(query_outputs, axis=0))

        return jnp.concatenate(outputs, axis=0)

    @staticmethod
    def _modulate_packed(
        x: jnp.ndarray,
        mod: jnp.ndarray,
        cu: jnp.ndarray,
        batch: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply (shift, scale, gate) modulation over the packed axis.

        Mirrors upstream ``_modulate(..., cu_lens=...)``: ``x`` is ``[1, N, dim]``,
        flattened to ``[N, dim]``, modulation parameters repeated per token by
        ``cu`` lengths, produces ``x * (1 + scale) + shift`` and the flat gate.
        """
        dim = mod.shape[-1] // 3
        xf = jnp.reshape(x, (-1, dim))
        lengths = cu[1:] - cu[:-1]
        sample_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), lengths)
        shift = mod[:, :dim]
        scale = mod[:, dim : 2 * dim]
        gate = mod[:, 2 * dim :]
        shift_t = shift[sample_ids]
        scale_t = scale[sample_ids]
        gate_t = gate[sample_ids]
        out = xf * (1.0 + scale_t) + shift_t
        return jnp.reshape(out, x.shape), gate_t

    # ------------------------------------------------------------------
    # Forward (documented mapping for the Keras ``call`` entrypoint).
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        image_rotary_emb: Any,
        txt_cu_lens: Any,
        img_cu_lens: Any,
        joint_attention_kwargs: dict[str, Any] | None = None,
        capture: bool = False,
    ) -> tuple[Any, Any] | tuple[Any, Any, dict[str, jnp.ndarray]]:
        """Execute Block-0 forward.

        Args:
            hidden_states: image stream, packed ``[1, N_img, dim]``.
            encoder_hidden_states: text stream, packed ``[1, N_txt, dim]``.
            temb: ``[B, dim]`` time/context embedding.
            image_rotary_emb: ``[N_img, head_dim//2, 2]`` cartesian (cos, sin).
            txt_cu_lens: ``[B+1]`` integer cumulative text lengths.
            img_cu_lens: ``[B+1]`` integer cumulative image lengths.
            joint_attention_kwargs: passed through for interface parity
                (attention semantics remain fixed: non-causal, dropout 0).
            capture: if True, also return an intermediates dict.

        Returns:
            ``(text_stream, image_stream)``, or with ``capture=True``
            ``(text_stream, image_stream, intermediates)``.
        """
        dim = self.dim
        hd = self.attention_head_dim
        heads = self.num_attention_heads

        joint_attention_kwargs = joint_attention_kwargs or {}
        if not isinstance(joint_attention_kwargs, dict):
            raise MageFlowBlock0Error("joint_attention_kwargs must be a dict or None")
        supported_attention_kwargs = {"attention_mode", "query_chunk"}
        unknown_attention_kwargs = sorted(
            set(joint_attention_kwargs) - supported_attention_kwargs
        )
        if unknown_attention_kwargs:
            raise MageFlowBlock0Error(
                "unsupported joint_attention_kwargs: "
                f"{unknown_attention_kwargs}; supported keys are "
                f"{sorted(supported_attention_kwargs)}"
            )
        attention_mode = joint_attention_kwargs.get(
            "attention_mode", "masked-global"
        )
        query_chunk = joint_attention_kwargs.get("query_chunk", 256)

        hs = jnp.asarray(hidden_states, dtype=jnp.float32)
        ens = jnp.asarray(encoder_hidden_states, dtype=jnp.float32)
        temb = jnp.asarray(temb, dtype=jnp.float32)
        freqs = jnp.asarray(image_rotary_emb, dtype=jnp.float32)

        if hs.ndim != 3 or hs.shape[-1] != dim:
            raise MageFlowBlock0Error(
                f"hidden_states must be [1, N_img, {dim}], got shape {hs.shape}"
            )
        if ens.ndim != 3 or ens.shape[-1] != dim:
            raise MageFlowBlock0Error(
                f"encoder_hidden_states must be [1, N_txt, {dim}], got shape {ens.shape}"
            )
        if hs.shape[0] != 1:
            raise MageFlowBlock0Error(
                f"packed image stream must have leading dim 1, got {hs.shape[0]} "
                "(cu_seqlens packing convention)"
            )
        if ens.shape[0] != 1:
            raise MageFlowBlock0Error(
                f"packed text stream must have leading dim 1, got {ens.shape[0]} "
                "(cu_seqlens packing convention)"
            )
        if temb.ndim != 2 or temb.shape[-1] != dim:
            raise MageFlowBlock0Error(
                f"temb must be [B, {dim}], got shape {temb.shape}"
            )

        n_img = int(hs.shape[1])
        n_txt = int(ens.shape[1])
        batch = int(temb.shape[0])

        txt_cu = _validate_cu_lens(txt_cu_lens, "txt_cu_lens", n_txt, batch)
        img_cu = _validate_cu_lens(img_cu_lens, "img_cu_lens", n_img, batch)

        if freqs.shape != (n_img, hd // 2, 2):
            raise MageFlowBlock0Error(
                f"image_rotary_emb must be [N_img={n_img}, head_dim//2={hd // 2}, 2], "
                f"got {freqs.shape}"
            )

        # ---- Modulation parameters (SiLU -> Linear(dim, 6*dim)). ---------
        img_mod = self._dense(jax.nn.silu(temb), f"{self._block_prefix}image_modulation")  # [B, 6d]
        txt_mod = self._dense(jax.nn.silu(temb), f"{self._block_prefix}text_modulation")
        img_mod1, img_mod2 = tuple(jnp.split(img_mod, 2, axis=-1))
        txt_mod1, txt_mod2 = tuple(jnp.split(txt_mod, 2, axis=-1))

        # ---- norm1 + modulation. ------------------------------------------
        img_normed = _layernorm(hs, self.eps)
        img_modulated, img_gate1 = self._modulate_packed(img_normed, img_mod1, img_cu, batch)
        txt_normed = _layernorm(ens, self.eps)
        txt_modulated, txt_gate1 = self._modulate_packed(txt_normed, txt_mod1, txt_cu, batch)

        # ---- Joint double-stream attention. --------------------------------
        # Image stream Q/K/V.
        img_q = jnp.reshape(self._dense(img_modulated, f"{self._block_prefix}attention.image_q"), (-1, heads, hd))
        img_k = jnp.reshape(self._dense(img_modulated, f"{self._block_prefix}attention.image_k"), (-1, heads, hd))
        img_v = jnp.reshape(self._dense(img_modulated, f"{self._block_prefix}attention.image_v"), (-1, heads, hd))
        # Text stream Q/K/V.
        txt_q = jnp.reshape(self._dense(txt_modulated, f"{self._block_prefix}attention.text_q"), (-1, heads, hd))
        txt_k = jnp.reshape(self._dense(txt_modulated, f"{self._block_prefix}attention.text_k"), (-1, heads, hd))
        txt_v = jnp.reshape(self._dense(txt_modulated, f"{self._block_prefix}attention.text_v"), (-1, heads, hd))

        # QK RMSNorm (both streams).
        img_q = _rms_norm(img_q, self.eps, jnp.asarray(self._param_paths[f"{self._block_prefix}attention.image_q_norm.scale"]))
        img_k = _rms_norm(img_k, self.eps, jnp.asarray(self._param_paths[f"{self._block_prefix}attention.image_k_norm.scale"]))
        txt_q = _rms_norm(txt_q, self.eps, jnp.asarray(self._param_paths[f"{self._block_prefix}attention.text_q_norm.scale"]))
        txt_k = _rms_norm(txt_k, self.eps, jnp.asarray(self._param_paths[f"{self._block_prefix}attention.text_k_norm.scale"]))

        # Image-only rotary embedding (text is never rotated).
        img_q = _apply_rotary(img_q, freqs)
        img_k = _apply_rotary(img_k, freqs)

        # Pack text-then-image and run segment-isolated attention.
        txt_dest, img_dest, joint_cu, _ = self._joint_pack_indices(
            txt_cu, img_cu, n_txt, n_img, batch
        )
        total = int(joint_cu[-1])
        joint_query = jnp.zeros((total, heads, hd), dtype=jnp.float32)
        joint_key = jnp.zeros((total, heads, hd), dtype=jnp.float32)
        joint_value = jnp.zeros((total, heads, hd), dtype=jnp.float32)
        joint_query = joint_query.at[txt_dest].set(txt_q)
        joint_query = joint_query.at[img_dest].set(img_q)
        joint_key = joint_key.at[txt_dest].set(txt_k)
        joint_key = joint_key.at[img_dest].set(img_k)
        joint_value = joint_value.at[txt_dest].set(txt_v)
        joint_value = joint_value.at[img_dest].set(img_v)

        joint_attn_out = self._segment_attention(
            joint_query,
            joint_key,
            joint_value,
            joint_cu,
            batch,
            attention_mode=attention_mode,
            query_chunk=query_chunk,
        )

        img_attn_flat = jnp.reshape(joint_attn_out[img_dest], (n_img, heads * hd))
        txt_attn_flat = jnp.reshape(joint_attn_out[txt_dest], (n_txt, heads * hd))

        # Output projections.
        img_attn_out = self._dense(img_attn_flat, f"{self._block_prefix}attention.image_out")  # [N_img, dim]
        txt_attn_out = jnp.reshape(
            self._dense(txt_attn_flat, f"{self._block_prefix}attention.text_out"), (1, n_txt, dim)
        )

        # Gated attention residuals (upstream: gate is per-token per-dim).
        hs = hs + img_gate1 * img_attn_out
        ens = ens + txt_gate1 * txt_attn_out

        post_attn_img = hs
        post_attn_txt = ens

        # ---- norm2 + modulation + MLP + gated residual. --------------------
        img_normed2 = _layernorm(hs, self.eps)
        img_modulated2, img_gate2 = self._modulate_packed(img_normed2, img_mod2, img_cu, batch)
        img_mlp_out = self._dense(
            _gelu_tanh(
                self._dense(img_modulated2, f"{self._block_prefix}image_mlp.gate_projection")
            ),
            f"{self._block_prefix}image_mlp.output_projection",
        )
        hs = hs + img_gate2 * img_mlp_out

        txt_normed2 = _layernorm(ens, self.eps)
        txt_modulated2, txt_gate2 = self._modulate_packed(txt_normed2, txt_mod2, txt_cu, batch)
        txt_mlp_out = self._dense(
            _gelu_tanh(
                self._dense(txt_modulated2, f"{self._block_prefix}text_mlp.gate_projection")
            ),
            f"{self._block_prefix}text_mlp.output_projection",
        )
        ens = ens + txt_gate2 * txt_mlp_out

        text_stream, image_stream = ens, hs
        if not capture:
            return text_stream, image_stream

        intermediates = {
            "image_q": img_q,
            "image_k": img_k,
            "image_v": img_v,
            "text_q": txt_q,
            "text_k": txt_k,
            "text_v": txt_v,
            "attention_image": img_attn_out,
            "attention_text": txt_attn_out,
            "post_attn_residual_image": post_attn_img,
            "post_attn_residual_text": post_attn_txt,
            "mlp_image": img_mlp_out,
            "mlp_text": txt_mlp_out,
            "output_image": image_stream,
            "output_text": text_stream,
        }
        return text_stream, image_stream, intermediates

    def call(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        image_rotary_emb: Any,
        txt_cu_lens: Any,
        img_cu_lens: Any,
        joint_attention_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Keras entrypoint; delegates to :meth:`forward`."""
        return self.forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            txt_cu_lens,
            img_cu_lens,
            joint_attention_kwargs=joint_attention_kwargs,
            capture=bool(kwargs.pop("capture", False)),
        )
