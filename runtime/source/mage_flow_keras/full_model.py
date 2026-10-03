"""G5: full Mage-Flow transformer (DiT) Keras/JAX target model.

Composes the validated Block-0 implementation (``MageFlowBlock0Candidate``)
into the complete ``MageFlow`` transformer geometry:

  * image input projection (``img_in``),
  * text RMSNorm + text input projection (``txt_norm`` / ``txt_in``),
  * time/text embedding (``time_text_embed.timestep_embedder``),
  * ``depth`` repeated dual-stream blocks (``transformer_blocks``),
  * adaptive final output norm (``norm_out.linear``) + output projection
    (``proj_out``).

Only the transformer (DiT) is in scope for the G5 static phase.  The Qwen3-VL
text encoder and the VAE are separate future component phases per
``PLANNED_KERAS_COMPONENT_SCHEMA_V1.md``.

Shape-only specification
------------------------

:func:`mage_flow_transformer_parameter_specs` returns the full parameter map
(path -> Keras shape) *without* allocating any weights.  The target parameter
inventory is generated from these specs (no 4.1B-parameter host allocation).

The Keras class owns the same parameter topology as the spec function; the
small-dimension regression test instantiates the class at reduced size and
proves ``parameter_shapes() == specs`` at that size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import keras

from mage_flow_keras.block0_candidate import (
    MageFlowBlock0Candidate,
    MageFlowBlock0Error,
    _rms_norm,
    block_parameter_leaf_shapes,
)

PERFECT_ROOT = "planned.mage_flow_transformer."
TIME_PROJ_CHANNELS = 256


@dataclass(frozen=True)
class MageFlowTransformerConfig:
    """Exact transformer (DiT) geometry used by the accepted mapping."""

    in_channels: int = 128
    out_channels: int = 128
    context_in_dim: int = 2560
    hidden_size: int = 3072
    num_heads: int = 24
    attention_head_dim: int = 128
    depth: int = 12
    eps: float = 1e-6
    axes_dim: tuple[int, int, int] = (16, 56, 56)

    def __post_init__(self) -> None:
        if self.num_heads * self.attention_head_dim != self.hidden_size:
            raise ValueError(
                "hidden_size must equal num_heads * attention_head_dim "
                f"({self.num_heads} * {self.attention_head_dim} != {self.hidden_size})"
            )
        if sum(self.axes_dim) != self.attention_head_dim:
            raise ValueError(
                f"axes_dim sum {sum(self.axes_dim)} must equal attention_head_dim "
                f"{self.attention_head_dim}"
            )


def _global_parameter_specs(
    config: MageFlowTransformerConfig,
) -> dict[str, tuple[int, ...]]:
    h = config.hidden_size
    return {
        f"{PERFECT_ROOT}image_input.kernel": (config.in_channels, h),
        f"{PERFECT_ROOT}image_input.bias": (h,),
        f"{PERFECT_ROOT}text_input_norm.scale": (config.context_in_dim,),
        f"{PERFECT_ROOT}text_input.kernel": (config.context_in_dim, h),
        f"{PERFECT_ROOT}text_input.bias": (h,),
        f"{PERFECT_ROOT}time_embedding.linear_1.kernel": (TIME_PROJ_CHANNELS, h),
        f"{PERFECT_ROOT}time_embedding.linear_1.bias": (h,),
        f"{PERFECT_ROOT}time_embedding.linear_2.kernel": (h, h),
        f"{PERFECT_ROOT}time_embedding.linear_2.bias": (h,),
        f"{PERFECT_ROOT}output_norm.modulation.kernel": (h, 2 * h),
        f"{PERFECT_ROOT}output_norm.modulation.bias": (2 * h,),
        f"{PERFECT_ROOT}output_projection.kernel": (h, config.out_channels),
        f"{PERFECT_ROOT}output_projection.bias": (config.out_channels,),
    }


def mage_flow_transformer_parameter_specs(
    config: MageFlowTransformerConfig | None = None,
) -> dict[str, tuple[int, ...]]:
    """Complete target parameter map (path -> Keras shape), allocation-free.

    Paths use the full ``planned.mage_flow_transformer.<...>`` namespace so the
    inventory joins directly with ``reference/transformer_mapping.json``
    ``target_path`` values.
    """
    config = config or MageFlowTransformerConfig()
    specs: dict[str, tuple[int, ...]] = {}
    specs.update(_global_parameter_specs(config))
    for index in range(config.depth):
        for leaf, shape in block_parameter_leaf_shapes(
            config.hidden_size, config.attention_head_dim
        ).items():
            specs[f"{PERFECT_ROOT}blocks.{index}.{leaf}"] = shape
    if len(specs) != config.depth * 32 + 13:
        raise AssertionError(
            "full target spec size mismatch: "
            f"{len(specs)} != depth({config.depth}) * 32 + 13"
        )
    return specs


class MageFlowTransformer(keras.Layer):
    """Full Mage-Flow transformer (DiT) as a Keras 3 / JAX layer.

    The same validated Block-0 math is reused for every block via
    ``MageFlowBlock0Candidate`` with a distinct ``block_index``.
    """

    def __init__(
        self,
        config: MageFlowTransformerConfig | None = None,
        name: str = "mage_flow_transformer",
    ):
        super().__init__(name=name, dtype="float32")
        if keras.config.backend() != "jax":
            raise MageFlowBlock0Error(
                "MageFlowTransformer requires the JAX backend "
                f"(KERAS_BACKEND=jax); active backend is {keras.config.backend()}"
            )
        self.config = config or MageFlowTransformerConfig()
        self._parameter_paths: dict[str, keras.Variable] = {}
        self._build_global_parameters()
        self._block_layers: dict[int, MageFlowBlock0Candidate] = {}
        for index in range(self.config.depth):
            self._block_layers[index] = MageFlowBlock0Candidate(
                dim=self.config.hidden_size,
                num_attention_heads=self.config.num_heads,
                attention_head_dim=self.config.attention_head_dim,
                eps=self.config.eps,
                block_index=index,
                name=f"mage_flow_block_{index}",
            )

    def _add_param(self, path: str, shape: tuple[int, ...]) -> keras.Variable:
        dense_name = path[len(PERFECT_ROOT):]
        var = self.add_weight(
            name=dense_name,
            shape=shape,
            initializer="zeros",
            trainable=True,
        )
        self._parameter_paths[path] = var
        return var

    def _build_global_parameters(self) -> None:
        for path, shape in _global_parameter_specs(self.config).items():
            self._add_param(path, shape)

    def parameter_paths(self) -> dict[str, keras.Variable]:
        """{full ``planned.mage_flow_transformer.<...>`` path: variable}."""
        paths: dict[str, keras.Variable] = dict(self._parameter_paths)
        for index in sorted(self._block_layers):
            for leaf, var in self._block_layers[index].parameter_paths().items():
                paths[f"{PERFECT_ROOT}{leaf}"] = var
        return paths

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {path: tuple(var.shape) for path, var in self.parameter_paths().items()}

    def _dense(self, x: jnp.ndarray, global_path: str) -> jnp.ndarray:
        kernel = jnp.asarray(
            self._parameter_paths[f"{PERFECT_ROOT}{global_path}.kernel"]
        )
        bias = jnp.asarray(self._parameter_paths[f"{PERFECT_ROOT}{global_path}.bias"])
        return jnp.matmul(x, kernel) + bias

    @classmethod
    def _time_text_embed(cls, timesteps: jnp.ndarray) -> jnp.ndarray:
        """Frozen sinusoidal time embedding (``MageFlowTimestepProjEmbeddings``).

        This G5 wiring mirrors the upstream vendored sinusoid
        (flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000, 256 channels).
        """
        half = TIME_PROJ_CHANNELS // 2
        exponent = -jnp.log(jnp.float32(10000.0)) * jnp.arange(
            0, half, step=1, dtype=jnp.float32
        ) / half
        emb_table = jnp.exp(exponent).astype(jnp.bfloat16)
        prod = timesteps[:, None].astype(jnp.float32) * emb_table[None, :]
        prod = 1000.0 * prod
        out = jnp.concatenate([jnp.sin(prod), jnp.cos(prod)], axis=-1)
        out = jnp.concatenate([out[:, half:], out[:, :half]], axis=-1)
        return out

    def forward(
        self,
        img: Any,
        txt: Any,
        timesteps: Any,
        image_rotary_emb: Any,
        img_cu_seqlens: Any,
        txt_cu_seqlens: Any,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the full transformer forward (G5 structural wiring).

        Mirrors ``MageFlow.forward`` ordering: image input projection, text
        RMSNorm + projection, sinusoidal time embedding through
        ``linear_1`` -> SiLU -> ``linear_2``, ``depth`` dual-stream blocks, the
        adaptive final output norm on the image stream, then the output
        projection.
        """
        dim = self.config.hidden_size
        img = jnp.asarray(img, dtype=jnp.float32)
        txt = jnp.asarray(txt, dtype=jnp.float32)
        timesteps = jnp.asarray(timesteps, dtype=jnp.bfloat16).astype(jnp.float32)

        image_stream = self._dense(img, "image_input")
        text_stream = _rms_norm(
            txt,
            self.config.eps,
            jnp.asarray(self._parameter_paths[f"{PERFECT_ROOT}text_input_norm.scale"]),
        )
        text_stream = self._dense(text_stream, "text_input")

        timesteps_proj = self._time_text_embed(timesteps)
        temb = self._dense(timesteps_proj, "time_embedding.linear_1")
        temb = self._dense(jax.nn.silu(temb), "time_embedding.linear_2")
        txt_vec = jnp.zeros((txt.shape[0], dim), dtype=jnp.float32)
        temb = temb + txt_vec

        for index in sorted(self._block_layers):
            block = self._block_layers[index]
            text_stream, image_stream = block.forward(
                hidden_states=image_stream,
                encoder_hidden_states=text_stream,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                txt_cu_lens=txt_cu_seqlens,
                img_cu_lens=img_cu_seqlens,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        image_stream = self._output_modulate(image_stream, temb, img_cu_seqlens)
        return self._dense(image_stream, "output_projection")

    def call(
        self,
        img: Any,
        txt: Any,
        timesteps: Any,
        image_rotary_emb: Any,
        img_cu_seqlens: Any,
        txt_cu_seqlens: Any,
        joint_attention_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        return self.forward(
            img,
            txt,
            timesteps,
            image_rotary_emb,
            img_cu_seqlens,
            txt_cu_seqlens,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    def _output_modulate(
        self,
        x: jnp.ndarray,
        conditioning: jnp.ndarray,
        cu_seqlens: Any,
    ) -> jnp.ndarray:
        """AdaLayerNormContinuous (structural): image stream only."""
        emb = self._dense(jax.nn.silu(conditioning), "output_norm.modulation")
        scale, shift = jnp.split(emb, 2, axis=-1)
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normed = (xf - mean) * jax.lax.rsqrt(var + self.config.eps)
        cu = jnp.asarray(cu_seqlens, dtype=jnp.int32)
        lengths = cu[1:] - cu[:-1]
        sample_ids = jnp.repeat(jnp.arange(conditioning.shape[0], dtype=jnp.int32), lengths)
        scale_t = scale[sample_ids]
        shift_t = shift[sample_ids]
        flat = jnp.reshape(normed, (-1, normed.shape[-1]))
        out = flat * (1.0 + scale_t) + shift_t
        return jnp.reshape(out, x.shape).astype(x.dtype)


CANONICAL_CONFIG = MageFlowTransformerConfig()
N_REPEATED_BLOCKS = CANONICAL_CONFIG.depth
GLOBAL_PARAMETER_COUNT = 13