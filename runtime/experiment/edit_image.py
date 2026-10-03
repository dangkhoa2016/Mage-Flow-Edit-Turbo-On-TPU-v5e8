"""Reference-image preprocessing helpers for Mage-Flow Edit."""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_reference(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Return RGB image as float32 NCHW in the closed range [-1, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
    return np.transpose(arr, (2, 0, 1))[None, ...]
