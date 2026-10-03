
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax


# =====================================================================
# Linear / Conv primitives
#
# Frozen checkpoint layout:
#
# Linear:
#   PyTorch [OUT, IN]
#   converted JAX [IN, OUT]
#
# Conv2d:
#   PyTorch [O, I, H, W]
#   converted JAX [H, W, I, O]
#
# Runtime activations remain NCHW.
# =====================================================================

def _linear(p, name, x):

    y = jnp.matmul(
        x,
        p[name + ".weight"],
    )

    bkey = name + ".bias"

    if bkey in p:
        y = y + p[bkey]

    return y


def _conv(
    p,
    name,
    x,
    *,
    stride=1,
    padding=0,
    groups=1,
):

    w = p[name + ".weight"]

    if isinstance(stride, int):
        stride = (
            stride,
            stride,
        )

    if isinstance(padding, int):
        padding = (
            (padding, padding),
            (padding, padding),
        )

    y = lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=stride,
        padding=padding,
        dimension_numbers=(
            "NCHW",
            "HWIO",
            "NCHW",
        ),
        feature_group_count=groups,
    )

    bkey = name + ".bias"

    if bkey in p:
        y = (
            y
            + p[bkey][
                None,
                :,
                None,
                None,
            ]
        )

    return y


# =====================================================================
# Activations / norms
# =====================================================================

def _silu(x):
    return x * jax.nn.sigmoid(x)


def _gelu(x):
    return jax.nn.gelu(
        x,
        approximate=False,
    )


def _layer_norm_2d(
    p,
    name,
    x,
    *,
    affine=True,
    eps=1e-6,
):

    dtype = x.dtype

    xf = x.astype(
        jnp.float32
    )

    mean = jnp.mean(
        xf,
        axis=1,
        keepdims=True,
    )

    var = jnp.mean(
        jnp.square(
            xf - mean
        ),
        axis=1,
        keepdims=True,
    )

    y = (
        (xf - mean)
        * lax.rsqrt(
            var + eps
        )
    ).astype(dtype)

    if affine:

        y = (
            y
            * p[
                name + ".weight"
            ][
                None,
                :,
                None,
                None,
            ]
            + p[
                name + ".bias"
            ][
                None,
                :,
                None,
                None,
            ]
        )

    return y


def _layer_norm_last(
    p,
    name,
    x,
    *,
    eps=1e-6,
):

    dtype = x.dtype

    xf = x.astype(
        jnp.float32
    )

    mean = jnp.mean(
        xf,
        axis=-1,
        keepdims=True,
    )

    var = jnp.mean(
        jnp.square(
            xf - mean
        ),
        axis=-1,
        keepdims=True,
    )

    y = (
        (xf - mean)
        * lax.rsqrt(
            var + eps
        )
    ).astype(dtype)

    return (
        y
        * p[name + ".weight"]
        + p[name + ".bias"]
    )


def _group_norm(
    p,
    name,
    x,
    *,
    groups=32,
    eps=1e-6,
):

    b, c, h, w = x.shape

    dtype = x.dtype

    y = x.reshape(
        b,
        groups,
        c // groups,
        h,
        w,
    ).astype(
        jnp.float32
    )

    mean = jnp.mean(
        y,
        axis=(2, 3, 4),
        keepdims=True,
    )

    var = jnp.mean(
        jnp.square(
            y - mean
        ),
        axis=(2, 3, 4),
        keepdims=True,
    )

    y = (
        (y - mean)
        * lax.rsqrt(
            var + eps
        )
    ).astype(dtype)

    y = y.reshape(
        b,
        c,
        h,
        w,
    )

    return (
        y
        * p[
            name + ".weight"
        ][
            None,
            :,
            None,
            None,
        ]
        + p[
            name + ".bias"
        ][
            None,
            :,
            None,
            None,
        ]
    )


def _rms_norm(
    p,
    name,
    x,
    *,
    eps=1e-6,
):

    dtype = x.dtype

    xf = x.astype(
        jnp.float32
    )

    var = jnp.mean(
        jnp.square(xf),
        axis=-1,
        keepdims=True,
    )

    y = (
        xf
        * lax.rsqrt(
            var + eps
        )
    ).astype(dtype)

    return (
        p[name + ".weight"]
        * y
    )


def _modulate_2d(
    x,
    shift,
    scale,
):

    return (
        x
        * (
            1
            + scale[
                :,
                :,
                None,
                None,
            ]
        )
        + shift[
            :,
            :,
            None,
            None,
        ]
    )


# =====================================================================
# TimestepEmbedder
# =====================================================================

def _timestep_embedding(
    t,
    dim,
    *,
    max_period=10000,
):

    half = dim // 2

    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(
            0,
            half,
            dtype=jnp.float32,
        )
        / half
    )

    args = (
        t.astype(
            jnp.float32
        )[:, None]
        * freqs[None, :]
    )

    emb = jnp.concatenate(
        [
            jnp.cos(args),
            jnp.sin(args),
        ],
        axis=-1,
    )

    if dim % 2:

        emb = jnp.concatenate(
            [
                emb,
                jnp.zeros_like(
                    emb[:, :1]
                ),
            ],
            axis=-1,
        )

    return emb


def _timestep_embedder(
    p,
    prefix,
    t,
):

    emb = _timestep_embedding(
        t,
        256,
    )

    emb = emb.astype(
        p[
            prefix
            + ".mlp.0.weight"
        ].dtype
    )

    x = _linear(
        p,
        prefix + ".mlp.0",
        emb,
    )

    x = _silu(x)

    return _linear(
        p,
        prefix + ".mlp.2",
        x,
    )


# =====================================================================
# DiCo blocks
# =====================================================================

def _dico_block(
    p,
    prefix,
    inp,
    c,
):

    modulation = _linear(
        p,
        prefix
        + ".adaLN_modulation.1",
        _silu(c),
    )

    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
    ) = jnp.split(
        modulation,
        6,
        axis=1,
    )

    x = _modulate_2d(
        _layer_norm_2d(
            p,
            prefix + ".norm1",
            inp,
            affine=False,
        ),
        shift_msa,
        scale_msa,
    )

    x = _conv(
        p,
        prefix + ".conv1",
        x,
    )

    x = _conv(
        p,
        prefix + ".conv2",
        x,
        padding=1,
        groups=x.shape[1],
    )

    x = _gelu(x)

    ca = jnp.mean(
        x,
        axis=(2, 3),
        keepdims=True,
    )

    ca = _conv(
        p,
        prefix + ".ca.1",
        ca,
    )

    ca = jax.nn.sigmoid(ca)

    x = x * ca

    x = _conv(
        p,
        prefix + ".conv3",
        x,
    )

    x = (
        inp
        + gate_msa[
            :,
            :,
            None,
            None,
        ]
        * x
    )

    y = _modulate_2d(
        _layer_norm_2d(
            p,
            prefix + ".norm2",
            x,
            affine=False,
        ),
        shift_mlp,
        scale_mlp,
    )

    y = _conv(
        p,
        prefix + ".conv4",
        y,
    )

    y = _gelu(y)

    y = _conv(
        p,
        prefix + ".conv5",
        y,
    )

    return (
        x
        + gate_mlp[
            :,
            :,
            None,
            None,
        ]
        * y
    )


def _encoder_dico_block(
    p,
    prefix,
    inp,
):

    x = _layer_norm_2d(
        p,
        prefix + ".norm1",
        inp,
        affine=True,
    )

    x = _conv(
        p,
        prefix + ".conv1",
        x,
    )

    x = _conv(
        p,
        prefix + ".conv2",
        x,
        padding=1,
        groups=x.shape[1],
    )

    x = _gelu(x)

    ca = jnp.mean(
        x,
        axis=(2, 3),
        keepdims=True,
    )

    ca = _conv(
        p,
        prefix + ".ca.1",
        ca,
    )

    ca = jax.nn.sigmoid(ca)

    x = x * ca

    x = _conv(
        p,
        prefix + ".conv3",
        x,
    )

    x = inp + x

    y = _layer_norm_2d(
        p,
        prefix + ".norm2",
        x,
        affine=True,
    )

    y = _conv(
        p,
        prefix + ".conv4",
        y,
    )

    y = _gelu(y)

    y = _conv(
        p,
        prefix + ".conv5",
        y,
    )

    return x + y


# =====================================================================
# Encoder
# =====================================================================

def _encoder_forward_pred(
    p,
    z_t,
    t,
    image,
):

    prefix = (
        "student.dconv_encoder"
    )

    cond = _conv(
        p,
        prefix
        + ".patch_cond_embed",
        image,
        stride=16,
    )

    for i in range(2):

        cond = _encoder_dico_block(
            p,
            f"{prefix}.head_blocks.{i}",
            cond,
        )

    cond = _conv(
        p,
        prefix + ".proj_down",
        cond,
    )

    zp = _conv(
        p,
        prefix + ".z_proj",
        z_t,
    )

    s = _conv(
        p,
        prefix + ".fuse_proj",
        jnp.concatenate(
            [
                cond,
                zp,
            ],
            axis=1,
        ),
    )

    c = _timestep_embedder(
        p,
        prefix + ".t_embedder",
        t,
    )

    for i in range(21):

        s = _dico_block(
            p,
            f"{prefix}.blocks.{i}",
            s,
            c,
        )

    s = _layer_norm_2d(
        p,
        prefix + ".norm_out",
        s,
        affine=True,
    )

    return _conv(
        p,
        prefix + ".proj_out",
        s,
    )


def encode_moments(
    p,
    image,
):

    b, _, h, w = image.shape

    if (
        h % 16
        or w % 16
    ):
        raise ValueError(
            "H,W must be multiples "
            "of 16"
        )

    z_t = jnp.zeros(
        (
            b,
            128,
            h // 16,
            w // 16,
        ),
        dtype=image.dtype,
    )

    t = jnp.zeros(
        (b,),
        dtype=image.dtype,
    )

    out = _encoder_forward_pred(
        p,
        z_t,
        t,
        image,
    )

    mean = out[:, :128]

    logvar = jnp.clip(
        out[:, 128:],
        -20.0,
        10.0,
    )

    return mean, logvar


def encode(
    p,
    image,
):

    mean, _ = encode_moments(
        p,
        image,
    )

    # Frozen model config:
    # sample_posterior = false
    return mean
