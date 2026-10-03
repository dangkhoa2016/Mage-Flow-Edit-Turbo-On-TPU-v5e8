"""JAX runtime for the Qwen3-VL visual tower used by Mage-Flow Edit."""
from __future__ import annotations

import numpy as np


def vision_position_ids(grid_thw, spatial_merge_size: int = 2):
    grid = np.asarray(grid_thw, dtype=np.int32)
    parts = []
    for t, h, w in grid.tolist():
        hp = np.arange(h, dtype=np.int32)[:, None].repeat(w, axis=1)
        hp = hp.reshape(h // spatial_merge_size, spatial_merge_size,
                        w // spatial_merge_size, spatial_merge_size)
        hp = hp.transpose(0, 2, 1, 3).reshape(-1)
        wp = np.arange(w, dtype=np.int32)[None, :].repeat(h, axis=0)
        wp = wp.reshape(h // spatial_merge_size, spatial_merge_size,
                        w // spatial_merge_size, spatial_merge_size)
        wp = wp.transpose(0, 2, 1, 3).reshape(-1)
        parts.append(np.stack([hp, wp], axis=-1).repeat(t, axis=0))
    return np.concatenate(parts, axis=0)


def vision_cu_seqlens(grid_thw):
    grid = np.asarray(grid_thw, dtype=np.int32)
    lengths = np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0])
    return np.concatenate([np.array([0], np.int32), np.cumsum(lengths, dtype=np.int32)])


def vision_bilinear_indices_and_weights(grid_thw, num_grid_per_side=48, spatial_merge_size=2):
    grid = np.asarray(grid_thw, dtype=np.int32)
    idx_parts = [[] for _ in range(4)]
    weight_parts = [[] for _ in range(4)]
    side = int(num_grid_per_side)
    m = int(spatial_merge_size)
    for t, h, w in grid.tolist():
        hg = np.linspace(0, side - 1, h, dtype=np.float32)
        wg = np.linspace(0, side - 1, w, dtype=np.float32)
        hf, wf = hg.astype(np.int32), wg.astype(np.int32)
        hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
        hfr, wfr = hg - hf, wg - wf
        corners = [
            (hf[:, None] * side + wf[None, :]).reshape(-1),
            (hf[:, None] * side + wc[None, :]).reshape(-1),
            (hc[:, None] * side + wf[None, :]).reshape(-1),
            (hc[:, None] * side + wc[None, :]).reshape(-1),
        ]
        weights = [
            ((1-hfr)[:,None]*(1-wfr)[None,:]).reshape(-1),
            ((1-hfr)[:,None]*wfr[None,:]).reshape(-1),
            (hfr[:,None]*(1-wfr)[None,:]).reshape(-1),
            (hfr[:,None]*wfr[None,:]).reshape(-1),
        ]
        hi = np.arange(h, dtype=np.int32).reshape(h // m, m)
        wi = np.arange(w, dtype=np.int32).reshape(w // m, m)
        reorder = (hi[:, :, None, None] * w + wi[None, None, :, :])
        reorder = reorder.transpose(0, 2, 1, 3).reshape(-1)
        reorder = np.tile(reorder, t)
        for i in range(4):
            idx_parts[i].append(corners[i][reorder])
            weight_parts[i].append(weights[i][reorder].astype(np.float32))
    indices = np.stack([np.concatenate(p) for p in idx_parts], axis=0)
    weights = np.stack([np.concatenate(p) for p in weight_parts], axis=0)
    return indices, weights


def _get(params, name):
    if name not in params:
        raise KeyError(f'missing visual parameter {name}')
    return params[name]


def _linear(params, prefix, x):
    import jax.numpy as jnp
    w = jnp.asarray(_get(params, prefix + '.kernel'))
    b = jnp.asarray(_get(params, prefix + '.bias'))
    return jnp.matmul(x, w) + b


def _layer_norm(params, prefix, x, eps=1e-6):
    import jax.numpy as jnp
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean((xf - mean) ** 2, axis=-1, keepdims=True)
    y = (xf - mean) * jax_lax_rsqrt(var + jnp.float32(eps))
    scale = jnp.asarray(_get(params, prefix + '.scale'), dtype=jnp.float32)
    bias = jnp.asarray(_get(params, prefix + '.bias'), dtype=jnp.float32)
    return (y * scale + bias).astype(x.dtype)


def jax_lax_rsqrt(x):
    import jax.lax as lax
    return lax.rsqrt(x)


def _gelu_tanh(x):
    import jax.numpy as jnp
    c = jnp.float32(0.7978845608028654)
    return jnp.float32(0.5) * x * (
        jnp.float32(1.0) + jnp.tanh(c * (x + jnp.float32(0.044715) * x**3))
    )


def _gelu_exact(x):
    import jax.numpy as jnp
    import jax.lax as lax
    return jnp.float32(0.5) * x * (
        jnp.float32(1.0) + lax.erf(x / jnp.sqrt(jnp.float32(2.0)))
    )


def patch_embed(params, pixel_values):
    import jax.numpy as jnp
    x = jnp.asarray(pixel_values, dtype=jnp.bfloat16)
    if x.ndim != 2 or x.shape[1] != 1536:
        raise ValueError(f'pixel_values must be [N,1536], got {x.shape}')
    x = x.reshape((-1, 3, 2, 16, 16)).transpose(0, 2, 3, 4, 1)
    w = jnp.asarray(_get(params, 'text_encoder.visual.patch_embed.proj.kernel'))
    b = jnp.asarray(_get(params, 'text_encoder.visual.patch_embed.proj.bias'))
    return jnp.tensordot(x, w, axes=((1,2,3,4),(0,1,2,3))) + b


def _vision_rotary(position_ids, dtype):
    import jax.numpy as jnp
    pos = jnp.asarray(position_ids, dtype=jnp.float32)
    inv = jnp.float32(1.0) / (
        jnp.float32(10000.0) ** (jnp.arange(0, 32, 2, dtype=jnp.float32) / jnp.float32(32.0))
    )
    rotary = (pos[..., None] * inv).reshape((pos.shape[0], -1))
    emb = jnp.concatenate([rotary, rotary], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x):
    import jax.numpy as jnp
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _apply_rotary(q, k, cos, sin):
    qf, kf = q.astype('float32'), k.astype('float32')
    cf, sf = cos.astype('float32')[:,None,:], sin.astype('float32')[:,None,:]
    qo = qf * cf + _rotate_half(qf) * sf
    ko = kf * cf + _rotate_half(kf) * sf
    return qo.astype(q.dtype), ko.astype(k.dtype)


def _attention(params, prefix, x, cos, sin, cu_seqlens):
    import jax.numpy as jnp
    qkv = _linear(params, prefix + '.qkv', x)
    n = qkv.shape[0]
    qkv = qkv.reshape(n, 3, 16, 64)
    q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]
    q, k = _apply_rotary(q, k, cos, sin)
    outputs = []
    bounds = np.asarray(cu_seqlens, dtype=np.int32).tolist()
    scale = jnp.float32(64.0 ** -0.5)
    for a, b in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = q[a:b], k[a:b], v[a:b]
        scores = jnp.einsum('qhd,khd->hqk', qs, ks).astype(jnp.float32) * scale
        probs = jax_softmax(scores, axis=-1).astype(q.dtype)
        out = jnp.einsum('hqk,khd->qhd', probs, vs).reshape((b-a, 1024))
        outputs.append(out)
    y = jnp.concatenate(outputs, axis=0)
    return _linear(params, prefix + '.proj', y)


def jax_softmax(x, axis=-1):
    import jax.nn as jnn
    return jnn.softmax(x, axis=axis)


def _mlp(params, prefix, x):
    h = _linear(params, prefix + '.linear_fc1', x)
    h = _gelu_tanh(h)
    return _linear(params, prefix + '.linear_fc2', h)


def _block(params, layer, x, cos, sin, cu_seqlens):
    base = f'text_encoder.visual.blocks.{layer}'
    h = _layer_norm(params, base + '.norm1', x)
    x = x + _attention(params, base + '.attn', h, cos, sin, cu_seqlens)
    h = _layer_norm(params, base + '.norm2', x)
    return x + _mlp(params, base + '.mlp', h)


def _merger(params, prefix, x, postshuffle_norm=False):
    import jax.numpy as jnp
    if postshuffle_norm:
        y = x.reshape((-1, 4096))
        y = _layer_norm(params, prefix + '.norm', y)
    else:
        y = _layer_norm(params, prefix + '.norm', x)
        y = y.reshape((-1, 4096))
    y = _linear(params, prefix + '.linear_fc1', y)
    y = _gelu_exact(y.astype(jnp.float32)).astype(y.dtype)
    return _linear(params, prefix + '.linear_fc2', y)


def _position_embeds(params, grid_thw):
    import jax.numpy as jnp
    idx, wt = vision_bilinear_indices_and_weights(grid_thw, 48, 2)
    table = jnp.asarray(_get(params, 'text_encoder.visual.pos_embed.embedding'))
    gathered = table[jnp.asarray(idx)]
    weights = jnp.asarray(wt, dtype=jnp.float32)[..., None]
    return jnp.sum(gathered.astype(jnp.float32) * weights, axis=0).astype(table.dtype)


def vision_forward(params, pixel_values, grid_thw):
    import jax.numpy as jnp
    grid = np.asarray(grid_thw, dtype=np.int32)
    x = patch_embed(params, pixel_values)
    pos = _position_embeds(params, grid)
    if x.shape != pos.shape:
        raise ValueError(f'patch/position shape mismatch {x.shape} != {pos.shape}')
    x = x + pos.astype(x.dtype)
    pids = vision_position_ids(grid, 2)
    cos, sin = _vision_rotary(pids, x.dtype)
    cu = vision_cu_seqlens(grid)
    deep = []
    deep_layers = (5, 11, 17)
    for layer in range(24):
        x = _block(params, layer, x, cos, sin, cu)
        if layer in deep_layers:
            i = deep_layers.index(layer)
            deep.append(_merger(params, f'text_encoder.visual.deepstack_merger_list.{i}', x, True))
    pool = _merger(params, 'text_encoder.visual.merger', x, False)
    return {'last_hidden_state': x, 'pooler_output': pool, 'deepstack_features': deep}
