# -*- coding: utf-8 -*-
"""image_postprocess — C7 final-image authority (C1R GAP-5 resolution).

scale/clip authority is MAGE_FLOW_G5_PIPELINE_INTEGRATION diagnostic (see root doc 04).
           MAGE_FLOW_G5_POST_VISUAL_FAILURE_DIAGNOSTIC.json + the R2 authority PNG.

Observed raw decode: shape [1,3,512,512] NCHW, range within [-1,1]
(``decoded_fraction_inside_minus1_plus1 = 1.0``, above/below = 0.0). The recorded PNG
channel statistics (mean [110.4,116.1,136.8], max [190,190,210], min [72,71,90]) are exactly
reproduced by ``x*127.5+127.5`` (=``(x+1)/2*255``) clipped to [0,255] and cast to uint8
(e.g. raw -0.4355 -> 71.98 -> 72, raw 0.6484 -> 210.2 -> 210). Output is RGB NHWC encoded
with Pillow.

Byte-exact historical PNG reproduction is NOT claimed (HISTORICAL_BASELINE_REPLAY_EXACT=False);
the Pillow encoder is a controlled constant shared by BOTH A/B paths.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .capture_hooks import CaptureHooks
from .runtime_contract import (
    FINAL_IMAGE_CHANNEL_ORDER,
    FINAL_IMAGE_CLIP,
    FINAL_IMAGE_DTYPE,
    FINAL_IMAGE_LAYOUT,
    FINAL_IMAGE_PNG_ENCODER,
    FINAL_IMAGE_SCALING,
    VAE_OUTPUT_LAYOUT,
)


def to_uint8_image(raw: Any) -> Any:
    """NCHW [-1,1] -> NHWC uint8 [0,255] (no silent dtype/scaling change of the compute path)."""
    import numpy as np

    a = np.asarray(raw)
    if a.ndim != 4 or a.shape[0] != 1:
        raise ValueError(f"expected NCHW [1,C,H,W]; got shape {a.shape}")
    if a.shape[1] != 3:
        raise ValueError(f"expected 3 channels; got {a.shape[1]}")
    nhwc = np.transpose(a[0], (1, 2, 0))  # HWC
    scaled = nhwc.astype(np.float64) * 127.5 + 127.5
    clipped = np.clip(scaled, 0.0, 255.0)
    return np.rint(clipped).astype(np.uint8)


def pixel_stats(image_u8: Any) -> dict:
    import numpy as np

    a = np.asarray(image_u8)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "channel_mean": [float(x) for x in a.reshape(-1, a.shape[-1]).mean(axis=0)],
        "channel_min": [int(x) for x in a.reshape(-1, a.shape[-1]).min(axis=0)],
        "channel_max": [int(x) for x in a.reshape(-1, a.shape[-1]).max(axis=0)],
    }


def encode_png(image_u8: Any) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.fromarray(image_u8, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def postprocess_and_capture(
    raw_vae_decode: Any,
    hooks: CaptureHooks,
    *,
    extra: dict | None = None,
) -> dict:
    """Full C7 chain: transform -> PNG -> stats -> hash -> hook. Returns the summary dict."""
    image_u8 = to_uint8_image(raw_vae_decode)
    stats = pixel_stats(image_u8)
    png_bytes = encode_png(image_u8)
    png_sha256 = hashlib.sha256(png_bytes).hexdigest()
    meta = {
        "vae_output_layout": VAE_OUTPUT_LAYOUT,
        "final_image_layout": FINAL_IMAGE_LAYOUT,
        "channel_order": FINAL_IMAGE_CHANNEL_ORDER,
        "scaling": FINAL_IMAGE_SCALING,
        "clip": FINAL_IMAGE_CLIP,
        "dtype": FINAL_IMAGE_DTYPE,
        "png_encoder": FINAL_IMAGE_PNG_ENCODER,
        "png_sha256": png_sha256,
        "pixel_stats": stats,
        "historical_baseline_replay_exact": False,
    }
    if extra:
        meta.update(extra)
    hooks.c7_final_image(png_bytes, stats, png_sha256)
    return {"image_uint8": image_u8, "png_sha256": png_sha256, "meta": meta}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import numpy as np

    # Authority-derived fixture: values at the recorded raw range endpoints map to 72 / 210.
    raw = np.zeros((1, 3, 2, 2), dtype=np.float32)
    raw[:, :, 0, 0] = -0.4355
    raw[:, :, 1, 1] = 0.6484
    img = to_uint8_image(raw)
    assert img.shape == (2, 2, 3), img.shape
    assert int(round(-0.4355 * 127.5 + 127.5)) == 72
    assert int(round(0.6484 * 127.5 + 127.5)) == 210
    assert img[0, 0, 0] == 72
    assert img[1, 1, 1] == 210
    with tempfile.TemporaryDirectory() as td:
        hooks = CaptureHooks(td)
        res = postprocess_and_capture(raw, hooks)
        assert Path(td, "final.png").exists()
        assert Path(td, "final.png.sha256").exists()
        print("C7_PNG_SHA256", res["png_sha256"][:16])
    print("IMAGE_POSTPROCESS_STATIC=PASS")
