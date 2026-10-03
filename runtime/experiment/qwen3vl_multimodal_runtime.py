"""Multimodal Qwen3-VL wrapper for Mage-Flow Edit JAX inference."""
from __future__ import annotations

import itertools
import numpy as np

IMAGE_TOKEN_ID = 151655


def _vision_llm_positions(start, grid, spatial_merge_size=2):
    t, h0, w0 = [int(x) for x in grid]
    h, w = h0 // spatial_merge_size, w0 // spatial_merge_size
    temporal = np.arange(t, dtype=np.int32)
    temporal = np.repeat(temporal, h * w) + start
    height = np.arange(h, dtype=np.int32) + start
    height = np.repeat(height, w)
    height = np.tile(height, t)
    width = np.arange(w, dtype=np.int32) + start
    width = np.tile(width, h * t)
    return np.stack([temporal, height, width], axis=0)


def multimodal_position_ids(input_ids, mm_token_type_ids, image_grid_thw,
                            spatial_merge_size=2, attention_mask=None):
    ids=np.asarray(input_ids,dtype=np.int32)
    types=np.asarray(mm_token_type_ids,dtype=np.int32)
    grid=np.asarray(image_grid_thw,dtype=np.int32)
    if ids.shape != types.shape:
        raise ValueError('input_ids/mm_token_type_ids shape mismatch')
    out=np.zeros((3,ids.shape[0],ids.shape[1]),dtype=np.int32)
    grid_iter=iter(grid.tolist())
    mask=np.ones_like(ids,dtype=bool) if attention_mask is None else np.asarray(attention_mask).astype(bool)
    for b in range(ids.shape[0]):
        valid_types=types[b][mask[b]]
        groups=[]
        for key, group in itertools.groupby(enumerate(valid_types.tolist()),lambda x:x[1]):
            g=list(group); groups.append((key,g[0][0],g[-1][0]+1))
        current=0; chunks=[]
        for typ,a,z in groups:
            if typ==0:
                n=z-a
                chunks.append(np.tile(np.arange(n,dtype=np.int32)[None,:]+current,(3,1)))
                current+=n
            elif typ==1:
                g=np.asarray(next(grid_iter),dtype=np.int32)
                chunks.append(_vision_llm_positions(current,g,spatial_merge_size))
                current+=max(int(g[1]),int(g[2]))//spatial_merge_size
            else:
                raise NotImplementedError('video tokens are not required for Mage-Flow Edit image mode')
        merged=np.concatenate(chunks,axis=1)
        out[:,b,mask[b]]=merged
    return out


def multimodal_language_forward(params, cfg, input_ids, position_ids,
                                image_embeds, deepstack_features):
    import jax.numpy as jnp
    from .text_encoder_runtime import (
        language_model_state, _get, _rotary_cos_sin,
        _decoder_layer, _rms_norm,
    )
    ids=jnp.asarray(input_ids,dtype=jnp.int32)
    if ids.ndim!=2 or ids.shape[0]!=1:
        raise ValueError(f'current Edit runtime requires batch=1, got {ids.shape}')
    lm=language_model_state(params,strict=False)
    base='text_encoder.language_model'
    embed=_get(lm,base+'.embed_tokens.embedding')
    x=embed[ids]
    image_embeds=jnp.asarray(image_embeds,dtype=x.dtype)
    n_image=image_embeds.shape[0]
    positions=jnp.nonzero(ids[0]==IMAGE_TOKEN_ID,size=n_image)[0]
    x=x.at[0,positions,:].set(image_embeds)
    pos=jnp.asarray(position_ids,dtype=jnp.int32)
    if pos.shape!=(3,1,ids.shape[1]):
        raise ValueError(f'position_ids shape {pos.shape} invalid')
    cos,sin=_rotary_cos_sin(cfg,pos,x.dtype)
    if len(deepstack_features)!=3:
        raise ValueError('Qwen3-VL Edit requires 3 DeepStack feature tensors')
    for layer in range(cfg.num_hidden_layers):
        x=_decoder_layer(lm,base,layer,x,cos,sin,cfg)
        if layer<3:
            d=jnp.asarray(deepstack_features[layer],dtype=x.dtype)
            if d.shape!=(n_image,cfg.hidden_size):
                raise ValueError(f'deepstack {layer} shape {d.shape} invalid')
            x=x.at[0,positions,:].add(d)
    x=_rms_norm(x,_get(lm,base+'.norm.scale'),cfg.rms_norm_eps)
    return x
