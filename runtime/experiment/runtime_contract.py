# -*- coding: utf-8 -*-
"""runtime_contract — immutable, frozen experiment/runtime configuration.

No value is ever obtained from stale notebook variables. The first deterministic case is
frozen from the durable preregistration (prior 10) and from the R2 authority JSON
(MAGE_FLOW_G5_PIPELINE_INTEGRATION_AUTHORITY_R2.json), replicated explicitly below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROPE_MODE_CONTROL = "control"
ROPE_MODE_DIAGNOSTIC = "c2a3-diagnostic"
VALID_ROPE_MODES = (ROPE_MODE_CONTROL, ROPE_MODE_DIAGNOSTIC)

EXPERIMENT_ID = "G5_T2I_BLUR_CAUSAL_CASE_001"
PROMPT = "A cinematic photograph of a red vintage car parked beside a mountain lake."
NEGATIVE_PROMPT = None
SEED = 42
HEIGHT = 512
WIDTH = 512
STEPS = 4
GUIDANCE = 1.0
STATIC_SHIFT = 6.0
SIGMA_SCHEDULE = [1.0, 0.9473684430122375, 0.8571428656578064, 0.6666666865348816, 0.0]
LATENT_SHAPE = [1, 128, 32, 32]
IMAGE_ROPE_SHAPE = [1024, 64, 2]

# Frozen choices where the historical authority is not byte-exact (prior preregistration 10).
INITIAL_NOISE_GENERATION_RULE = "jax.random.normal(PRNGKey(seed), float32 [1,128,32,32])"
VAE_SCALING_RULE = "IDENTITY"

NEW_DETERMINISTIC_BASELINE = True
HISTORICAL_BASELINE_REPLAY_EXACT = False

# ---------------------------------------------------------------------------
# Text Encoder tokenization contract (C1R-GAP-1 resolution).
#
# Authority: installed `diffusers` QwenImagePipeline (the Mage-Flow input model is a
# diffusers-style repo with text_encoder = transformers Qwen3VLForConditionalGeneration and
# tokenizer = transformers AutoProcessor; model_index.json). The QwenImage pipeline defines
#     prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the
#         color, shape, size, texture, quantity, text, spatial relationships of the objects
#         and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
#     prompt_template_encode_start_idx = 34
# and slices the last hidden state as ``e[drop_idx:]`` after tokenizing the templated prompt.
#
# Cross-validation against durable R2 authority
# (MAGE_FLOW_G5_PIPELINE_INTEGRATION_AUTHORITY_R2.json):
#     prompt_input_tokens = 53        -> tokenized template length (verified this session: 53)
#     prompt_drop_idx     = 34        -> prompt_template_encode_start_idx
#     prompt_conditioning_tokens = 19 -> 53 - 34 (verified this session: 19)
# Verified with the model's own tokenizer.json (no transformers import required).
# ---------------------------------------------------------------------------
TE_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
TE_PROMPT_DROP_IDX = 34
TE_PROMPT_INPUT_TOKENS = 53
TE_PROMPT_CONDITIONING_TOKENS = 19
TE_TOKENIZER_ASSET = "tokenizer.json"

# ---------------------------------------------------------------------------
# C2 Transformer-boundary contract (runbook sections 14-15). The selected boundary is the
# block-0 attention-adjacent tensor already exposed by the durable runtime:
# `block0_candidate.py::MageFlowBlock0Candidate.forward(..., capture=True)` returns
# ``(text_stream, image_stream, intermediates)`` and `intermediates` already contains the
# post-RoPE image query. Capture is a test-only subclass (never edits production source).
# ---------------------------------------------------------------------------
C2_SYMBOL = "mage_flow_keras.block0_candidate.MageFlowBlock0Candidate.forward"
C2_FILE = "block0_candidate.py"
C2_BOUNDARY_DESCRIPTION = (
    "block-0 image-stream query tensor after image RoPE (post-_apply_rotary), exposed via "
    "the durable capture=True return path; also persists the block-0 image-stream output"
)
C2_EXPECTED_DTYPE = "float32"
# [N_img=latent_h*latent_w=1024, num_attention_heads=24, attention_head_dim=128]
C2_EXPECTED_SHAPE = [1024, 24, 128]
C2_CAPTURE_MECHANISM = (
    "test-only MageFlowTransformer subclass forcing capture=(index==0) and invoking an "
    "explicit observer callback; production source is never patched"
)

# ---------------------------------------------------------------------------
# C7 final-image postprocess contract (runbook section 16).
# Authority: MAGE_FLOW_G5_POST_VISUAL_FAILURE_DIAGNOSTIC.json + R2 authority PNG.
#   raw_vae_decode shape [1,3,512,512] (NCHW), range inside [-1,1]
#   (decoded_fraction_inside_minus1_plus1 = 1.0; above_plus_1 = 0.0; below_minus_1 = 0.0)
#   PNG channel stats reproduce x*127.5+127.5 clipped to [0,255] (raw min -0.4355 -> 72,
#   raw max 0.6484 -> 210, matching png_channel_min=72 / png_channel_max=210).
# ---------------------------------------------------------------------------
VAE_OUTPUT_LAYOUT = "NCHW"
FINAL_IMAGE_LAYOUT = "NHWC"
FINAL_IMAGE_CHANNEL_ORDER = "RGB"
FINAL_IMAGE_SCALING = "x*127.5+127.5  (== (x+1)/2*255)"
FINAL_IMAGE_CLIP = "[0,255]"
FINAL_IMAGE_DTYPE = "uint8"
FINAL_IMAGE_PNG_ENCODER = "Pillow PIL.Image.fromarray(..., mode='RGB')"


@dataclass(frozen=True)
class ResolvedArtifacts:
    """Snapshot of an artifact resolution needed by the runtime contract."""

    source_root: Path
    model_root: Path
    text_encoder_checkpoint: Optional[Path]
    transformer_checkpoint: Optional[Path]
    vae_checkpoint: Optional[Path]
    rope_fixture: Path
    basis_dim16: Path
    basis_dim56: Path
    output_root: Path
    text_encoder_runtime_module: Optional[Path] = None
    transformer_runtime_source: Optional[Path] = None
    vae_runtime_module: Optional[Path] = None


@dataclass(frozen=True)
class ExperimentContract:
    """Immutable runtime contract for one controlled A/B case."""

    experiment_id: str = EXPERIMENT_ID
    prompt: str = PROMPT
    negative_prompt: Optional[str] = NEGATIVE_PROMPT
    seed: int = SEED
    height: int = HEIGHT
    width: int = WIDTH
    steps: int = STEPS
    guidance: float = GUIDANCE
    static_shift: float = STATIC_SHIFT
    sigma_schedule: list = field(default_factory=lambda: list(SIGMA_SCHEDULE))
    latent_shape: list = field(default_factory=lambda: list(LATENT_SHAPE))
    rope_mode: str = ROPE_MODE_CONTROL
    output_dir: str = ""
    text_encoder_checkpoint: str = ""
    transformer_checkpoint: str = ""
    vae_checkpoint: str = ""
    source_hashes: dict = field(default_factory=dict)
    new_deterministic_baseline: bool = NEW_DETERMINISTIC_BASELINE
    historical_baseline_replay_exact: bool = HISTORICAL_BASELINE_REPLAY_EXACT

    def __post_init__(self) -> None:
        if self.rope_mode not in VALID_ROPE_MODES:
            raise ValueError(f"unknown rope_mode {self.rope_mode!r} (must be in {VALID_ROPE_MODES})")

    @property
    def is_diagnostic(self) -> bool:
        return self.rope_mode == ROPE_MODE_DIAGNOSTIC

    def diagnostic_only(self) -> bool:
        return self.is_diagnostic

    def to_banner(self) -> str:
        return "\n".join(
            [
                f"EXPERIMENT_ID={self.experiment_id}",
                f"ROPE_MODE={self.rope_mode.upper()}",
                f"DIAGNOSTIC_ONLY={self.is_diagnostic}",
                "PRODUCTION_DEFAULT_CHANGED=False",
                "HISTORICAL_PROVENANCE_CLAIM=False",
                f"PROMPT={self.prompt}",
                f"SEED={self.seed}",
                f"RESOLUTION={self.width}x{self.height}",
                f"STEPS={self.steps}",
                f"GUIDANCE={self.guidance}",
                f"SIGMA_SCHEDULE={self.sigma_schedule}",
                f"LATENT_SHAPE={self.latent_shape}",
                f"VAE_SCALING_RULE={VAE_SCALING_RULE}",
                f"INITIAL_NOISE_GENERATION_RULE={INITIAL_NOISE_GENERATION_RULE}",
            ]
        )

    def to_json(self) -> str:
        import json

        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "guidance": self.guidance,
            "static_shift": self.static_shift,
            "sigma_schedule": self.sigma_schedule,
            "latent_shape": self.latent_shape,
            "rope_mode": self.rope_mode,
            "output_dir": self.output_dir,
            "text_encoder_checkpoint": self.text_encoder_checkpoint,
            "transformer_checkpoint": self.transformer_checkpoint,
            "vae_checkpoint": self.vae_checkpoint,
            "source_hashes": self.source_hashes,
            "new_deterministic_baseline": self.new_deterministic_baseline,
            "historical_baseline_replay_exact": self.historical_baseline_replay_exact,
        }


def contract_with_mode(base: ExperimentContract, mode: str, output_dir: str) -> ExperimentContract:
    """Return an equivalent contract with only rope_mode/output_dir changed (A/B isolation)."""
    if mode not in VALID_ROPE_MODES:
        raise ValueError(f"unknown rope_mode {mode!r}")
    return ExperimentContract(
        experiment_id=base.experiment_id,
        prompt=base.prompt,
        negative_prompt=base.negative_prompt,
        seed=base.seed,
        height=base.height,
        width=base.width,
        steps=base.steps,
        guidance=base.guidance,
        static_shift=base.static_shift,
        sigma_schedule=list(base.sigma_schedule),
        latent_shape=list(base.latent_shape),
        rope_mode=mode,
        output_dir=output_dir,
        text_encoder_checkpoint=base.text_encoder_checkpoint,
        transformer_checkpoint=base.transformer_checkpoint,
        vae_checkpoint=base.vae_checkpoint,
        source_hashes=dict(base.source_hashes),
        new_deterministic_baseline=base.new_deterministic_baseline,
        historical_baseline_replay_exact=base.historical_baseline_replay_exact,
    )


def equivalence_except_rope(a: ExperimentContract, b: ExperimentContract) -> list[str]:
    """A/B config-equivalence check. Returns list of differing fields (empty => equivalent).

    Controlled-variable isolation: every field must be identical except rope_mode and the
    diagnostic labels derived from it. output_dir is allowed to differ (run directory).
    """
    differences: list[str] = []
    for key in {
        "experiment_id", "prompt", "negative_prompt", "seed", "height", "width",
        "steps", "guidance", "static_shift", "sigma_schedule", "latent_shape",
        "text_encoder_checkpoint", "transformer_checkpoint", "vae_checkpoint",
    }:
        if a.as_dict()[key] != b.as_dict()[key]:
            differences.append(key)
    if (a.new_deterministic_baseline != b.new_deterministic_baseline) or (
        a.historical_baseline_replay_exact != b.historical_baseline_replay_exact
    ):
        differences.append("baseline_flags")
    return differences


if __name__ == "__main__":
    c = ExperimentContract()
    print(c.to_banner())
    print("EQUIVALENCE_SELF_TEST")
    a = contract_with_mode(c, ROPE_MODE_CONTROL, "runs/A_control/run_001")
    b = contract_with_mode(c, ROPE_MODE_DIAGNOSTIC, "runs/B_c2a3_diagnostic/run_001")
    print("MODE_A", a.rope_mode, "DIAGNOSTIC_ONLY", a.diagnostic_only())
    print("MODE_B", b.rope_mode, "DIAGNOSTIC_ONLY", b.diagnostic_only())
    print("DIFFERENCES", equivalence_except_rope(a, b))
    assert a.diagnostic_only() is False
    assert b.diagnostic_only() is True
    assert equivalence_except_rope(a, b) == []
    print("RUNTIME_CONTRACT_VALIDATION=PASS")