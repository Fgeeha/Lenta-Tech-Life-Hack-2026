"""Optional RealESRGAN 4x super-resolution for price-tag crops.

Activated via SHELF_SR_ENABLED=1.  Requires:
  - models/RealESRGAN_x4plus.pth (already present)
  - realesrgan + basicsr packages (with torchvision compat shim)

The shim patches torchvision.transforms.functional_tensor which was removed
in torchvision 0.17+ but is still imported by basicsr 1.x.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path("models/RealESRGAN_x4plus.pth")
_upsampler: object | None = None
_load_attempted: bool = False


def _apply_tv_shim() -> None:
    """Patch torchvision.transforms.functional_tensor for torchvision ≥ 0.17."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as _F

        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = _F.rgb_to_grayscale  # type: ignore[attr-defined]
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    except Exception:
        pass


def _load_upsampler() -> object | None:
    global _upsampler, _load_attempted
    if _load_attempted:
        return _upsampler
    _load_attempted = True

    model_path = Path(os.getenv("SHELF_SR_MODEL", str(_DEFAULT_MODEL)))
    if not model_path.exists():
        logger.debug("SR model not found at %s — SR disabled", model_path)
        return None

    try:
        _apply_tv_shim()
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda"),
            device=device,
        )
        logger.info("RealESRGAN 4x loaded on %s (model=%s)", device, model_path)
        return _upsampler
    except Exception as exc:
        logger.warning("Could not load RealESRGAN: %s", exc)
        return None


def is_available() -> bool:
    """Return True if the SR model file exists."""
    return Path(os.getenv("SHELF_SR_MODEL", str(_DEFAULT_MODEL))).exists()


def is_enabled() -> bool:
    """Return True when SHELF_SR_ENABLED=1 and model is available."""
    return os.getenv("SHELF_SR_ENABLED", "").lower() in ("1", "true", "yes") and is_available()


def upscale_crop(crop: np.ndarray) -> np.ndarray:
    """Apply 4x ESRGAN super-resolution.  Returns the original on any failure."""
    if crop is None or crop.size == 0:
        return crop

    upsampler = _load_upsampler()
    if upsampler is None:
        return crop

    try:
        import cv2

        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        output, _ = upsampler.enhance(crop, outscale=4)  # type: ignore[attr-defined]
        return output
    except Exception as exc:
        logger.warning("SR enhance failed: %s", exc)
        return crop
