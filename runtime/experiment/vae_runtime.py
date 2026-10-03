
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


# =====================================================================
# CoD latent decoder
# =====================================================================

def _resnet_block(
    p,
    prefix,
    x,
):

    h = _group_norm(
        p,
        prefix + ".norm1",
        x,
    )

    h = _silu(h)

    h = _conv(
        p,
        prefix + ".conv1",
        h,
        padding=1,
    )

    h = _group_norm(
        p,
        prefix + ".norm2",
        h,
    )

    h = _silu(h)

    h = _conv(
        p,
        prefix + ".conv2",
        h,
        padding=1,
    )

    shortcut = (
        prefix
        + ".nin_shortcut.weight"
    )

    if shortcut in p:

        x = _conv(
            p,
            prefix + ".nin_shortcut",
            x,
        )

    return x + h


def _attention_block(
    p,
    prefix,
    x,
    *,
    patch_size=32,
):

    h0 = _group_norm(
        p,
        prefix + ".norm",
        x,
    )

    Q = _conv(
        p,
        prefix + ".q",
        h0,
    )

    K = _conv(
        p,
        prefix + ".k",
        h0,
    )

    V = _conv(
        p,
        prefix + ".v",
        h0,
    )

    d = patch_size

    b, c, H, W = Q.shape

    pad_h = (
        d - H % d
    ) % d

    pad_w = (
        d - W % d
    ) % d

    if pad_h or pad_w:

        pads = (
            (0, 0),
            (0, 0),
            (0, pad_h),
            (0, pad_w),
        )

        Q = jnp.pad(
            Q,
            pads,
            mode="edge",
        )

        K = jnp.pad(
            K,
            pads,
            mode="edge",
        )

        V = jnp.pad(
            V,
            pads,
            mode="edge",
        )

    H_pad = Q.shape[2]
    W_pad = Q.shape[3]

    nph = H_pad // d
    npw = W_pad // d

    npatches = (
        nph * npw
    )

    def to_patches(t):

        return (
            t.reshape(
                b,
                c,
                nph,
                d,
                npw,
                d,
            )
            .transpose(
                0,
                2,
                4,
                1,
                3,
                5,
            )
            .reshape(
                b * npatches,
                c,
                d * d,
            )
        )

    Qp = to_patches(Q)
    Kp = to_patches(K)
    Vp = to_patches(V)

    attn = (
        jnp.einsum(
            "bci,bcj->bij",
            Qp,
            Kp,
        )
        * (c ** -0.5)
    )

    attn = jax.nn.softmax(
        attn,
        axis=2,
    )

    attn = attn.transpose(
        0,
        2,
        1,
    )

    out = jnp.einsum(
        "bcj,bji->bci",
        Vp,
        attn,
    )

    out = (
        out.reshape(
            b,
            nph,
            npw,
            c,
            d,
            d,
        )
        .transpose(
            0,
            3,
            1,
            4,
            2,
            5,
        )
        .reshape(
            b,
            c,
            H_pad,
            W_pad,
        )
    )

    if pad_h or pad_w:

        out = out[
            :,
            :,
            :H,
            :W,
        ]

    out = _conv(
        p,
        prefix + ".proj_out",
        out,
    )

    return x + out


def _decoder_condition(
    p,
    z,
):

    prefix = (
        "pipeline."
        "y_embedder.decoder"
    )

    h = _conv(
        p,
        prefix + ".conv_in",
        z,
        padding=1,
    )

    h = _resnet_block(
        p,
        prefix + ".block.0",
        h,
    )

    h = _attention_block(
        p,
        prefix + ".block.1",
        h,
        patch_size=32,
    )

    h = _resnet_block(
        p,
        prefix + ".block.2",
        h,
    )

    h = _attention_block(
        p,
        prefix + ".block.3",
        h,
        patch_size=32,
    )

    h = _resnet_block(
        p,
        prefix + ".block.4",
        h,
    )

    h = _group_norm(
        p,
        prefix + ".norm_out",
        h,
    )

    h = _silu(h)

    return _conv(
        p,
        prefix + ".conv_out",
        h,
        padding=1,
    )


# =====================================================================
# unfold/fold equivalent for kernel=stride=16
# =====================================================================

def _unfold_nonoverlap(
    x,
    *,
    ps=16,
):

    b, c, H, W = x.shape

    gh = H // ps
    gw = W // ps

    return (
        x.reshape(
            b,
            c,
            gh,
            ps,
            gw,
            ps,
        )
        .transpose(
            0,
            1,
            3,
            5,
            2,
            4,
        )
        .reshape(
            b,
            c * ps * ps,
            gh * gw,
        )
    )


def _fold_nonoverlap(
    patches,
    *,
    batch,
    gh,
    gw,
    ps=16,
):

    # patches:
    # [B*L, P2, C]

    channels = (
        patches.shape[-1]
    )

    x = patches.reshape(
        batch,
        gh * gw,
        ps * ps,
        channels,
    )

    x = x.transpose(
        0,
        1,
        3,
        2,
    )

    x = x.reshape(
        batch,
        gh,
        gw,
        channels,
        ps,
        ps,
    )

    x = x.transpose(
        0,
        3,
        1,
        4,
        2,
        5,
    )

    return x.reshape(
        batch,
        channels,
        gh * ps,
        gw * ps,
    )


# =====================================================================
# NerfEmbedder
# =====================================================================

def _nerf_embed(
    p,
    prefix,
    x,
    *,
    max_freqs=8,
):

    _, P2, _ = x.shape

    ps = int(
        P2 ** 0.5
    )

    dtype = x.dtype

    pos = jnp.linspace(
        0,
        1,
        ps,
        dtype=dtype,
    )

    pos_y, pos_x = (
        jnp.meshgrid(
            pos,
            pos,
            indexing="ij",
        )
    )

    pos_x = pos_x.reshape(
        -1,
        1,
        1,
    )

    pos_y = pos_y.reshape(
        -1,
        1,
        1,
    )

    freqs = jnp.linspace(
        0,
        max_freqs,
        max_freqs,
        dtype=dtype,
    )

    fx = freqs[
        None,
        :,
        None,
    ]

    fy = freqs[
        None,
        None,
        :,
    ]

    coeffs = (
        1
        + fx * fy
    ) ** -1

    dct_x = jnp.cos(
        pos_x
        * fx
        * jnp.pi
    )

    dct_y = jnp.cos(
        pos_y
        * fy
        * jnp.pi
    )

    dct = (
        dct_x
        * dct_y
        * coeffs
    ).reshape(
        1,
        -1,
        max_freqs ** 2,
    )

    dct = jnp.broadcast_to(
        dct,
        (
            x.shape[0],
            P2,
            max_freqs ** 2,
        ),
    )

    return _linear(
        p,
        prefix + ".embedder.0",
        jnp.concatenate(
            [
                x,
                dct,
            ],
            axis=-1,
        ),
    )


# =====================================================================
# Final decoder MLP
# =====================================================================

def _mlp_res_block(
    p,
    prefix,
    x,
    y,
):

    shift_scale_gate = (
        _linear(
            p,
            prefix
            + ".adaLN_modulation.1",
            _silu(y),
        )
    )

    (
        shift,
        scale,
        gate,
    ) = jnp.split(
        shift_scale_gate,
        3,
        axis=-1,
    )

    h = _layer_norm_last(
        p,
        prefix + ".in_ln",
        x,
    )

    h = (
        h
        * (1 + scale)
        + shift
    )

    h = _linear(
        p,
        prefix + ".mlp.0",
        h,
    )

    h = _silu(h)

    h = _linear(
        p,
        prefix + ".mlp.2",
        h,
    )

    return (
        x
        + gate * h
    )


def _decode_denoiser(
    p,
    noise,
    t,
    cond,
):

    prefix = "pipeline"

    b, _, H, W = (
        noise.shape
    )

    c = _timestep_embedder(
        p,
        prefix + ".t_embedder",
        t,
    )

    s1 = _conv(
        p,
        prefix
        + ".s_embedder.proj1",
        noise,
        stride=16,
    )

    s = _conv(
        p,
        prefix
        + ".s_embedder.proj2",
        jnp.concatenate(
            [
                s1,
                cond,
            ],
            axis=1,
        ),
    )

    for i in range(21):

        s = _dico_block(
            p,
            f"{prefix}.blocks.{i}",
            s,
            c,
        )

    gh = s.shape[2]
    gw = s.shape[3]

    length = (
        gh * gw
    )

    # source:
    # s.permute(0,2,3,1)
    #  .reshape(-1, hidden_size)

    sflat = (
        s.transpose(
            0,
            2,
            3,
            1,
        )
        .reshape(
            b * length,
            s.shape[1],
        )
    )

    # torch.nn.functional.unfold(
    #   x,
    #   kernel_size=16,
    #   stride=16
    # )

    x_unfold = (
        _unfold_nonoverlap(
            noise,
            ps=16,
        )
    )

    y = _conv(
        p,
        prefix
        + ".y_embedder_x",
        cond,
    )

    y_flat = y.reshape(
        b,
        y.shape[1],
        length,
    )

    # [B, 768+8192, L]

    x = jnp.concatenate(
        [
            x_unfold,
            y_flat,
        ],
        axis=1,
    )

    # source:
    #
    # x.reshape(
    #   b,
    #   -1,
    #   patch_size ** 2,
    #   length
    # )
    # .permute(0,3,2,1)
    # .flatten(0,1)

    x = (
        x.reshape(
            b,
            -1,
            16 * 16,
            length,
        )
        .transpose(
            0,
            3,
            2,
            1,
        )
        .reshape(
            b * length,
            16 * 16,
            -1,
        )
    )

    x = _nerf_embed(
        p,
        prefix + ".x_embedder",
        x,
        max_freqs=8,
    )

    # SimpleMLPAdaLN.input_proj

    x = _linear(
        p,
        prefix
        + ".dec_net.input_proj",
        x,
    )

    # c:
    # [B*L, 384]
    #
    # cond_embed:
    # 384 -> 256 * 32

    cond_tokens = _linear(
        p,
        prefix
        + ".dec_net.cond_embed",
        sflat,
    )

    cond_tokens = (
        cond_tokens.reshape(
            b * length,
            16 * 16,
            -1,
        )
    )

    for i in range(3):

        x = _mlp_res_block(
            p,
            f"{prefix}."
            f"dec_net.res_blocks.{i}",
            x,
            cond_tokens,
        )

    # NerfFinalLayer

    x = _rms_norm(
        p,
        prefix
        + ".final_layer.norm",
        x,
    )

    x = _linear(
        p,
        prefix
        + ".final_layer.linear",
        x,
    )

    # x:
    # [B*L, 256, 3]
    #
    # source then performs:
    # transpose(1,2)
    # reshape(B,L,-1)
    # transpose(1,2)
    # F.fold

    return _fold_nonoverlap(
        x,
        batch=b,
        gh=gh,
        gw=gw,
        ps=16,
    )


def decode(
    p,
    latent,
):

    cond = _decoder_condition(
        p,
        latent,
    )

    b = latent.shape[0]

    H = (
        latent.shape[2]
        * 16
    )

    W = (
        latent.shape[3]
        * 16
    )

    noise = jnp.zeros(
        (
            b,
            3,
            H,
            W,
        ),
        dtype=latent.dtype,
    )

    t = jnp.zeros(
        (b,),
        dtype=latent.dtype,
    )

    return _decode_denoiser(
        p,
        noise,
        t,
        cond,
    )
