"""Real-ESRGAN x4plus super-resolution for tiny price-tag ROIs.

Controlled by SHELF_USE_SR=true/false (default: false).
Downloads weights on first use (~64 MB, cached in models/).
Applied ONLY to small ROIs (QR zone ~20-30 px, price zone ~80 px)
to keep inference time acceptable on CPU (≈0.5-2 s per tiny ROI).

On HF Spaces free tier (CPU basic): set SHELF_USE_SR=false (default).
For local inference with CPU: SHELF_USE_SR=true gives +quality on QR/price.
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SR_ENABLED = os.environ.get("SHELF_USE_SR", "false").lower() == "true"
_WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "RealESRGAN_x4plus.pth"
_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
_MODEL = None


def is_enabled() -> bool:
    return _SR_ENABLED


def _ensure_weights() -> bool:
    if _WEIGHTS_PATH.exists():
        return True
    logger.info("Downloading RealESRGAN weights (~64MB) to %s", _WEIGHTS_PATH)
    try:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_WEIGHTS_URL, str(_WEIGHTS_PATH))
        size_mb = _WEIGHTS_PATH.stat().st_size / 1e6
        logger.info("RealESRGAN weights downloaded: %.1f MB", size_mb)
        return True
    except Exception as exc:
        logger.warning("Cannot download RealESRGAN weights: %s", exc)
        return False


def _patch_torchvision() -> None:
    """Shim for torchvision ≥0.17 which removed functional_tensor submodule."""
    import sys
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import types
            import torchvision.transforms.functional as _F
            _mod = types.ModuleType("torchvision.transforms.functional_tensor")
            for _attr in dir(_F):
                setattr(_mod, _attr, getattr(_F, _attr))
            sys.modules["torchvision.transforms.functional_tensor"] = _mod
        except Exception:
            pass


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not _ensure_weights():
        return None
    try:
        _patch_torchvision()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        _MODEL = RealESRGANer(
            scale=4,
            model_path=str(_WEIGHTS_PATH),
            model=net,
            tile=0,  # no tiling — ROIs are tiny (<128 px)
            tile_pad=10,
            pre_pad=0,
            half=False,  # CPU inference: no FP16
            device="cpu",
        )
        logger.info("RealESRGAN model loaded (CPU)")
        return _MODEL
    except Exception as exc:
        logger.warning("Cannot load RealESRGAN model: %s", exc)
        return None


def upscale_roi(img: np.ndarray, outscale: int = 4) -> np.ndarray:
    """Super-resolve a tiny ROI with Real-ESRGAN x4plus.

    Falls back to cv2.INTER_LANCZOS4 if SR is disabled or model unavailable.

    Args:
        img: Small BGR image (e.g., 20×20 QR patch, 80×60 price zone).
        outscale: Target upscale factor (4 for the x4plus model).

    Returns:
        Upscaled BGR image at outscale× resolution.
    """
    if img is None or img.size == 0:
        return img

    def _fallback() -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * outscale, h * outscale), interpolation=cv2.INTER_LANCZOS4
        )

    if not _SR_ENABLED:
        return _fallback()

    model = _load_model()
    if model is None:
        return _fallback()

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output, _ = model.enhance(img_rgb, outscale=outscale)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.debug("RealESRGAN enhance failed (%s), using fallback", exc)
        return _fallback()
