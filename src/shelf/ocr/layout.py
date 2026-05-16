"""Rule-based Lenta price-tag layout priors.

The challenge template materials show that most tags keep semantic zones in
stable relative positions: product name in the upper part, QR on the right or
right-bottom side, barcode/SKU/date close to the lower band, and large prices in
the middle/lower price area.  This module turns those priors into safe ROI boxes
that improve OCR/QR/barcode attempts without replacing the generic fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import cv2
import numpy as np


class TemplateType(str, Enum):
    """Coarse Lenta price-tag mechanics used by downstream priors."""

    REGULAR = "regular"
    PROMO = "promo"
    DISCOUNT = "discount"
    WHOLESALE = "wholesale"
    BOGOF = "bogof"
    UNKNOWN = "unknown"


class Orientation(str, Enum):
    """Approximate crop orientation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ROTATED = "rotated"


@dataclass(frozen=True)
class LayoutROI:
    """One ROI in both relative and absolute coordinates.

    Relative coordinates are normalized to [0, 1] in the oriented crop frame.
    ``purpose`` is a broad semantic group such as ``qr`` or ``price``.
    """

    name: str
    purpose: str
    x0: float
    y0: float
    x1: float
    y1: float

    def clipped(self) -> "LayoutROI":
        """Return a copy with coordinates clipped to valid normalized bounds."""
        x0 = min(1.0, max(0.0, self.x0))
        y0 = min(1.0, max(0.0, self.y0))
        x1 = min(1.0, max(0.0, self.x1))
        y1 = min(1.0, max(0.0, self.y1))
        if x1 <= x0:
            x1 = min(1.0, x0 + 0.01)
        if y1 <= y0:
            y1 = min(1.0, y0 + 0.01)
        return LayoutROI(self.name, self.purpose, x0, y0, x1, y1)

    def as_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert the ROI to pixel coordinates inside a crop."""
        roi = self.clipped()
        w, h = max(1, int(width)), max(1, int(height))
        x0 = int(round(roi.x0 * w))
        y0 = int(round(roi.y0 * h))
        x1 = int(round(roi.x1 * w))
        y1 = int(round(roi.y1 * h))
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


@dataclass(frozen=True)
class LayoutPrior:
    """A coarse price-tag layout prediction with semantic ROIs."""

    template_type: TemplateType
    orientation: Orientation
    rois: dict[str, LayoutROI]

    def by_purpose(self, purpose: str) -> list[LayoutROI]:
        """Return all ROIs matching a semantic purpose."""
        return [roi for roi in self.rois.values() if roi.purpose == purpose]


_PROMO_RE = re.compile(r"акци|скид|выгод|карт", re.IGNORECASE)
_WHOLESALE_RE = re.compile(r"\bот\s*\d+|опт|порог|до\s*\d+", re.IGNORECASE)
_BOGOF_RE = re.compile(r"bogof|богоф|набор|2\s*цена|цена\s*2", re.IGNORECASE)
_DISCOUNT_RE = re.compile(r"[-−–]\s*\d{1,3}\s*(?:%|р|руб)", re.IGNORECASE)


def classify_orientation(crop: np.ndarray | None) -> Orientation:
    """Classify crop orientation from aspect ratio.

    Very narrow/tall crops are treated as vertical; very wide crops as
    horizontal.  Ambiguous crops are considered rotated because many shelf videos
    contain 90-degree oriented tags.
    """
    if crop is None or crop.size == 0:
        return Orientation.HORIZONTAL
    h, w = crop.shape[:2]
    if w >= h * 1.18:
        return Orientation.HORIZONTAL
    if h >= w * 1.18:
        return Orientation.VERTICAL
    return Orientation.ROTATED


def classify_template(
    crop: np.ndarray | None = None, texts: Iterable[str] | None = None
) -> TemplateType:
    """Classify a tag into a coarse mechanics bucket.

    The function is intentionally conservative: text hints win when available;
    otherwise color cues detect promo/discount tags.  Unknown falls back to the
    regular layout zones downstream.
    """
    joined = (
        " ".join(str(t or "") for t in (texts or [])).lower().replace("ё", "е")
    )
    if _BOGOF_RE.search(joined):
        return TemplateType.BOGOF
    if _WHOLESALE_RE.search(joined):
        return TemplateType.WHOLESALE
    if _DISCOUNT_RE.search(joined):
        return TemplateType.DISCOUNT
    if _PROMO_RE.search(joined):
        return TemplateType.PROMO

    if crop is None or crop.size == 0:
        return TemplateType.UNKNOWN
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 50, 60]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 50, 60]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([8, 45, 70]), np.array([42, 255, 255]))
    red_frac = float(((red1 | red2).mean()) / 255.0)
    orange_frac = float(orange.mean() / 255.0)
    if red_frac > 0.035:
        return TemplateType.DISCOUNT
    if orange_frac > 0.07:
        return TemplateType.PROMO
    return TemplateType.REGULAR


def get_layout_prior(
    crop: np.ndarray | None = None, texts: Iterable[str] | None = None
) -> LayoutPrior:
    """Return template/orientation-aware semantic ROIs for a crop.

    ROIs are broad by design: they should increase the chance of reading tiny QR
    and text without excluding fallback full-crop OCR/decoding.
    """
    orientation = classify_orientation(crop)
    template = classify_template(crop, texts)
    rois = _base_rois(orientation)
    rois.update(_template_overrides(template, orientation))
    rois = {name: roi.clipped() for name, roi in rois.items()}
    return LayoutPrior(
        template_type=template, orientation=orientation, rois=rois
    )


def roi_crops(
    crop: np.ndarray,
    purposes: set[str] | None = None,
    *,
    min_size: int = 12,
) -> list[tuple[str, np.ndarray]]:
    """Extract named crops for requested purposes from a price-tag crop."""
    if crop is None or crop.size == 0:
        return []
    prior = get_layout_prior(crop)
    h, w = crop.shape[:2]
    out: list[tuple[str, np.ndarray]] = []
    for name, roi in prior.rois.items():
        if purposes is not None and roi.purpose not in purposes:
            continue
        x0, y0, x1, y1 = roi.as_pixels(w, h)
        if x1 - x0 < min_size or y1 - y0 < min_size:
            continue
        out.append((name, crop[y0:y1, x0:x1]))
    return out


def _base_rois(orientation: Orientation) -> dict[str, LayoutROI]:
    if orientation == Orientation.VERTICAL:
        return {
            "name_roi": LayoutROI("name_roi", "name", 0.02, 0.02, 0.98, 0.34),
            "qr_roi": LayoutROI("qr_roi", "qr", 0.58, 0.58, 0.99, 0.99),
            "qr_top_roi": LayoutROI("qr_top_roi", "qr", 0.58, 0.00, 0.99, 0.34),
            "barcode_roi": LayoutROI(
                "barcode_roi", "barcode", 0.10, 0.72, 0.92, 0.99
            ),
            "price_card_roi": LayoutROI(
                "price_card_roi", "price", 0.05, 0.43, 0.96, 0.82
            ),
            "price_default_roi": LayoutROI(
                "price_default_roi", "price", 0.05, 0.31, 0.96, 0.58
            ),
            "sku_roi": LayoutROI("sku_roi", "sku", 0.02, 0.68, 0.72, 0.93),
            "datetime_roi": LayoutROI(
                "datetime_roi", "datetime", 0.02, 0.82, 0.72, 0.99
            ),
            "code_roi": LayoutROI("code_roi", "code", 0.02, 0.56, 0.78, 0.78),
            "symbol_roi": LayoutROI(
                "symbol_roi", "symbol", 0.00, 0.30, 0.38, 0.58
            ),
            "info_roi": LayoutROI("info_roi", "info", 0.02, 0.25, 0.98, 0.55),
        }
    if orientation == Orientation.ROTATED:
        return {
            "name_roi": LayoutROI("name_roi", "name", 0.02, 0.02, 0.98, 0.42),
            "qr_roi": LayoutROI("qr_roi", "qr", 0.58, 0.42, 0.99, 0.99),
            "barcode_roi": LayoutROI(
                "barcode_roi", "barcode", 0.08, 0.62, 0.94, 0.99
            ),
            "price_card_roi": LayoutROI(
                "price_card_roi", "price", 0.18, 0.43, 0.98, 0.93
            ),
            "price_default_roi": LayoutROI(
                "price_default_roi", "price", 0.18, 0.28, 0.96, 0.63
            ),
            "sku_roi": LayoutROI("sku_roi", "sku", 0.02, 0.55, 0.72, 0.85),
            "datetime_roi": LayoutROI(
                "datetime_roi", "datetime", 0.02, 0.76, 0.72, 0.99
            ),
            "code_roi": LayoutROI("code_roi", "code", 0.02, 0.47, 0.75, 0.70),
            "symbol_roi": LayoutROI(
                "symbol_roi", "symbol", 0.00, 0.28, 0.35, 0.58
            ),
            "info_roi": LayoutROI("info_roi", "info", 0.02, 0.22, 0.98, 0.55),
        }
    return {
        "name_roi": LayoutROI("name_roi", "name", 0.02, 0.02, 0.72, 0.42),
        "name_wide_roi": LayoutROI(
            "name_wide_roi", "name", 0.02, 0.02, 0.98, 0.34
        ),
        "qr_roi": LayoutROI("qr_roi", "qr", 0.68, 0.02, 0.99, 0.55),
        "qr_bottom_right_roi": LayoutROI(
            "qr_bottom_right_roi", "qr", 0.66, 0.48, 0.99, 0.99
        ),
        "barcode_roi": LayoutROI(
            "barcode_roi", "barcode", 0.18, 0.68, 0.96, 0.99
        ),
        "barcode_bottom_right_roi": LayoutROI(
            "barcode_bottom_right_roi", "barcode", 0.48, 0.62, 0.99, 0.99
        ),
        "price_card_roi": LayoutROI(
            "price_card_roi", "price", 0.38, 0.42, 0.98, 0.96
        ),
        "price_default_roi": LayoutROI(
            "price_default_roi", "price", 0.33, 0.28, 0.98, 0.60
        ),
        "sku_roi": LayoutROI("sku_roi", "sku", 0.02, 0.62, 0.62, 0.94),
        "datetime_roi": LayoutROI(
            "datetime_roi", "datetime", 0.02, 0.78, 0.64, 0.99
        ),
        "code_roi": LayoutROI("code_roi", "code", 0.02, 0.50, 0.64, 0.75),
        "symbol_roi": LayoutROI("symbol_roi", "symbol", 0.00, 0.34, 0.24, 0.66),
        "info_roi": LayoutROI("info_roi", "info", 0.02, 0.30, 0.62, 0.66),
    }


def _template_overrides(
    template: TemplateType, orientation: Orientation
) -> dict[str, LayoutROI]:
    # Discount/promo tags often have the discount badge on the left and QR on
    # the lower-right, so protect those areas with dedicated broad ROIs.
    if template in {TemplateType.PROMO, TemplateType.DISCOUNT}:
        if orientation == Orientation.VERTICAL:
            return {
                "discount_badge_roi": LayoutROI(
                    "discount_badge_roi", "discount", 0.00, 0.00, 0.48, 0.28
                ),
                "promo_price_roi": LayoutROI(
                    "promo_price_roi", "price", 0.02, 0.35, 0.98, 0.90
                ),
            }
        return {
            "discount_badge_roi": LayoutROI(
                "discount_badge_roi", "discount", 0.00, 0.00, 0.32, 0.45
            ),
            "promo_price_roi": LayoutROI(
                "promo_price_roi", "price", 0.25, 0.32, 0.98, 0.96
            ),
        }
    if template == TemplateType.WHOLESALE:
        return {
            "wholesale_info_roi": LayoutROI(
                "wholesale_info_roi", "info", 0.00, 0.35, 0.55, 0.75
            ),
            "wholesale_price_roi": LayoutROI(
                "wholesale_price_roi", "price", 0.28, 0.40, 0.98, 0.97
            ),
        }
    if template == TemplateType.BOGOF:
        return {
            "bogof_info_roi": LayoutROI(
                "bogof_info_roi", "info", 0.00, 0.25, 0.98, 0.75
            ),
            "bogof_price_roi": LayoutROI(
                "bogof_price_roi", "price", 0.28, 0.45, 0.98, 0.98
            ),
        }
    return {}
