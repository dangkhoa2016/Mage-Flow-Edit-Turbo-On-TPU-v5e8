# -*- coding: utf-8 -*-
"""text_encoder_driver — bound-runtime driver for the Text Encoder.

C1R GAP-1 (tokenization) and GAP-2 (TE runtime source not durable) are both RESOLVED:
the runtime now lives in-package (``text_encoder_runtime.py``, durable + self-contained) and
the tokenization semantics come from the durable installed diffusers QwenImage template +
drop index, cross-validated against R2 authority (53 -> 34 -> 19).

Entry points:
  restore_text_encoder(...)        -> TextEncoderHandle (config + tokenizer path + runtime)
  tokenize_prompt(...)             -> int32 ids of the templated prompt (host, no weights)
  run_text_encoder(handle, ids)    -> full TE output + conditioning slice + C0 capture
  encode_text(handle, texts, ...)  -> tokenize + forward (execute_compute=True)

No CUDA dependency. No CPU fallback for the TPU main forward. JAX is imported lazily inside
``text_encoder_runtime``. Tokenization uses the ``tokenizers`` library directly (the model's
own tokenizer.json) rather than ``transformers``, whose import performs a TPU metadata probe
that fails in a CPU-only session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .capture_hooks import CaptureHooks
from .runtime_contract import (
    TE_PROMPT_CONDITIONING_TOKENS,
    TE_PROMPT_DROP_IDX,
    TE_PROMPT_INPUT_TOKENS,
    TE_PROMPT_TEMPLATE,
    TE_TOKENIZER_ASSET,
)
from .text_encoder_runtime import (
    TE_CHECKPOINT_AUTHORITY,
    TE_DTYPE,
    TE_LANGUAGE_MODEL_LEAF_COUNT,
    TE_RUNTIME_SOURCE_DURABLE,
    TE_TOTAL_LEAF_COUNT,
    TextEncoderConfig,
    TextEncoderRuntime,
    bind_text_encoder_state,
    build_text_encoder_runtime,
    config_from_model_root,
    restore_text_encoder_checkpoint,
    text_encoder_forward,
)


@dataclass
class TextEncoderHandle:
    """Handle returned by restore_text_encoder."""

    config: TextEncoderConfig
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    runtime: TextEncoderRuntime = field(default_factory=TextEncoderRuntime)
    model_root: Optional[str] = None
    dtype: str = TE_DTYPE

    def assert_config_contract(self) -> None:
        self.runtime.assert_config_contract()

    def validate_checkpoint_metadata(self) -> None:
        if self.checkpoint_path:
            cp = Path(self.checkpoint_path)
            if not cp.exists():
                raise FileNotFoundError(
                    "TE checkpoint bytes not present at "
                    f"{self.checkpoint_path} (TEXT_ENCODER_CHECKPOINT_BYTES_PRESENT=False; "
                    "Stage D must provide/serve them before compute)"
                )
            present = [p.name for p in cp.iterdir()] if cp.is_dir() else []
            if present and len(present) != TE_CHECKPOINT_AUTHORITY["file_count"]:
                raise AssertionError(
                    f"TE checkpoint file count {len(present)} != "
                    f"{TE_CHECKPOINT_AUTHORITY['file_count']}"
                )


def _resolve_tokenizer_path(model_root: Optional[str]) -> Optional[str]:
    if not model_root:
        return None
    root = Path(model_root)
    candidates = [root / TE_TOKENIZER_ASSET, root / "text_encoder" / TE_TOKENIZER_ASSET]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def restore_text_encoder(
    *,
    text_encoder_checkpoint: Optional[str] = None,
    model_root: Optional[str] = None,
    config: Optional[TextEncoderConfig] = None,
    mesh: Any = None,
    params: Optional[dict] = None,
    execute_compute: bool = False,
) -> TextEncoderHandle:
    """Build a TE handle: config + tokenizer path + (optionally bound) runtime.

    ``execute_compute=False`` (preflight) validates config/tokenizer/checkpoint authority and
    does not restore weights. ``execute_compute=True`` restores + binds the converted
    checkpoint (requires durable bytes; fails closed otherwise).
    """
    if config is None and model_root:
        config = config_from_model_root(model_root)
    handle = TextEncoderHandle(
        config=config if config is not None else TextEncoderConfig(),
        checkpoint_path=text_encoder_checkpoint,
        tokenizer_path=_resolve_tokenizer_path(model_root),
        model_root=model_root,
    )
    if model_root:
        handle.assert_config_contract()

    state = params
    if execute_compute and state is None:
        if not text_encoder_checkpoint:
            raise RuntimeError(
                "text_encoder_checkpoint is required when execute_compute=True "
                "(failing closed; no placeholder weights)"
            )
        state = restore_text_encoder_checkpoint(
            text_encoder_checkpoint, mesh=mesh, expected_leaf_count=TE_TOTAL_LEAF_COUNT
        )
    handle.runtime = build_text_encoder_runtime(config=handle.config, params=state)
    return handle


def _load_tokenizer(tokenizer_path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def tokenize_prompt(prompt: str, tokenizer_path: str) -> dict:
    """Tokenize one prompt through the frozen QwenImage template. Host-only, no weights."""
    tok = _load_tokenizer(tokenizer_path)
    templated = TE_PROMPT_TEMPLATE.format(prompt)
    ids = tok.encode(templated).ids
    return {
        "template": TE_PROMPT_TEMPLATE,
        "templated_text": templated,
        "input_ids": [int(i) for i in ids],
        "input_dtype": "int32",
        "total_tokens": len(ids),
        "drop_idx": TE_PROMPT_DROP_IDX,
        "conditioning_tokens": len(ids) - TE_PROMPT_DROP_IDX,
    }


def tokenization_contract(token_info: dict) -> dict:
    return {
        "provider": "diffusers.QwenImagePipeline.prompt_template_encode + tokenizers.Tokenizer",
        "drop_idx": token_info["drop_idx"],
        "total_tokens": token_info["total_tokens"],
        "conditioning_tokens": token_info["conditioning_tokens"],
        "expected_total_tokens": TE_PROMPT_INPUT_TOKENS,
        "expected_conditioning_tokens": TE_PROMPT_CONDITIONING_TOKENS,
        "runtime_total": TE_PROMPT_INPUT_TOKENS,
        "runtime_conditioning": TE_PROMPT_CONDITIONING_TOKENS,
    }


def run_text_encoder(
    handle: TextEncoderHandle,
    input_ids: Any,
    hooks: Optional[CaptureHooks] = None,
) -> dict:
    """Execution edge: bound TE forward -> full output + conditioning slice (drops 34)."""
    if handle.runtime is None or not handle.runtime.bound:
        raise RuntimeError(
            "run_text_encoder requires a bound runtime "
            "(restore with execute_compute=True; failing closed)"
        )
    full = text_encoder_forward(handle.runtime, input_ids)
    total = int(full.shape[1])
    if total <= TE_PROMPT_DROP_IDX:
        raise ValueError(f"TE sequence length {total} <= drop_idx {TE_PROMPT_DROP_IDX}")
    conditioning = full[:, TE_PROMPT_DROP_IDX:, :]
    extra = {
        "full_output_shape": list(full.shape),
        "conditioning_shape": list(conditioning.shape),
        "drop_idx": TE_PROMPT_DROP_IDX,
        "conditioning_tokens": int(conditioning.shape[1]),
    }
    if hooks is not None:
        hooks.c0_text_encoder_output(full, extra=extra)
    return {
        "tensor": conditioning,
        "full_output": full,
        "shape": list(conditioning.shape),
        "full_output_shape": list(full.shape),
        "drop_idx": TE_PROMPT_DROP_IDX,
        "extra": extra,
    }


def encode_text(
    handle: TextEncoderHandle,
    texts: list[str],
    hooks: Optional[CaptureHooks] = None,
    *,
    execute_compute: bool = False,
) -> dict:
    """Tokenize and (if computing) forward the Text Encoder.

    ``execute_compute=False`` returns the tokenization contract without running model compute
    and without writing a fabricated C0 tensor. ``execute_compute=True`` runs the bound
    runtime and captures C0 = full TE output.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    if handle.tokenizer_path is None:
        raise FileNotFoundError(
            "tokenizer asset not found (expected <model_root>/text_encoder/tokenizer.json)"
        )
    infos = [tokenize_prompt(t, handle.tokenizer_path) for t in texts]
    contract = {
        "tokenization": [tokenization_contract(i) for i in infos],
        "framework": handle.runtime.model_kind,
        "dtype": handle.dtype,
        "execute_compute": execute_compute,
    }
    if not execute_compute:
        return {"compute": "DEFERRED", "token_contract": contract}

    import jax.numpy as jnp

    outputs = []
    for info in infos:
        ids = jnp.asarray(info["input_ids"], dtype=jnp.int32)[None, :]
        outputs.append(run_text_encoder(handle, ids, hooks=hooks))
    return {"compute": "EXECUTED", "token_contract": contract, "outputs": outputs}


if __name__ == "__main__":
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import tempfile

    import numpy as np

    from .text_encoder_runtime import TextEncoderConfig

    model_root = str(Path(__file__).resolve().parents[2])

    # 1) Real tokenizer contract from the durable model assets (host-only, no jax).
    tok_path = _resolve_tokenizer_path(model_root)
    assert tok_path, "model tokenizer.json not found"
    info = tokenize_prompt("a test prompt", tok_path)
    print("TE_TOKENIZER_PATH", tok_path)
    print("TOKENIZE_TOTAL", info["total_tokens"], "DROP", info["drop_idx"],
          "COND", info["conditioning_tokens"])

    # 2) Execution edge on a tiny synthetic bound runtime (real JAX, CPU backend).
    cfg = TextEncoderConfig(
        hidden_size=32, num_hidden_layers=2, vocab_size=64, head_dim=8,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
        rms_norm_eps=1e-6, rope_theta=1e4, mrope_section=[2, 1, 1],
    )
    rng = np.random.RandomState(7)
    pref = "text_encoder.language_model"
    p = {f"{pref}.embed_tokens.embedding": (rng.randn(cfg.vocab_size, cfg.hidden_size) * 0.05).astype(np.float32)}
    for i in range(cfg.num_hidden_layers):
        b = f"{pref}.layers.{i}"
        p[f"{b}.input_layernorm.scale"] = np.ones(cfg.hidden_size, np.float32)
        p[f"{b}.post_attention_layernorm.scale"] = np.ones(cfg.hidden_size, np.float32)
        p[f"{b}.self_attn.q_proj.kernel"] = (rng.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim) * 0.05).astype(np.float32)
        p[f"{b}.self_attn.k_proj.kernel"] = (rng.randn(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim) * 0.05).astype(np.float32)
        p[f"{b}.self_attn.v_proj.kernel"] = (rng.randn(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim) * 0.05).astype(np.float32)
        p[f"{b}.self_attn.o_proj.kernel"] = (rng.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size) * 0.05).astype(np.float32)
        p[f"{b}.self_attn.q_norm.scale"] = np.ones(cfg.head_dim, np.float32)
        p[f"{b}.self_attn.k_norm.scale"] = np.ones(cfg.head_dim, np.float32)
        p[f"{b}.mlp.gate_proj.kernel"] = (rng.randn(cfg.hidden_size, cfg.intermediate_size) * 0.05).astype(np.float32)
        p[f"{b}.mlp.up_proj.kernel"] = (rng.randn(cfg.hidden_size, cfg.intermediate_size) * 0.05).astype(np.float32)
        p[f"{b}.mlp.down_proj.kernel"] = (rng.randn(cfg.intermediate_size, cfg.hidden_size) * 0.05).astype(np.float32)
    p[f"{pref}.norm.scale"] = np.ones(cfg.hidden_size, np.float32)

    handle = restore_text_encoder(config=cfg, params=p)
    with tempfile.TemporaryDirectory() as td:
        hooks = CaptureHooks(td)
        ids = np.arange(1, 41, dtype=np.int32)[None, :]  # 40 tokens > drop_idx
        res = run_text_encoder(handle, ids, hooks=hooks)
        assert res["shape"] == [1, 40 - TE_PROMPT_DROP_IDX, cfg.hidden_size], res["shape"]
        assert Path(td, "C0_text_encoder_output.json").exists()
        assert Path(td, "C0_text_encoder_output.npy").exists()
        print("TE_EXECUTION_EDGE=PASS", res["full_output_shape"], "->", res["shape"])
    print("TEXT_ENCODER_RUNTIME_SOURCE_DURABLE", TE_RUNTIME_SOURCE_DURABLE)
    print("TE_LANGUAGE_MODEL_LEAF_COUNT", TE_LANGUAGE_MODEL_LEAF_COUNT)
    print("TEXT_ENCODER_DRIVER_STATIC=PASS")
