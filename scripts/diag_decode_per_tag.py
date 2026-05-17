"""Decode trace + ROI quality diagnostic (A1 + A2).

For each GT tag: tries fast decoders at the GT frame and ±100/200/300ms
offsets. Records per-frame sharpness, raw decode strings (before EAN-13
validation), and categorises WHY undecoded crops fail.

WeChatQR is intentionally skipped (2-second timeout per call would take
1918 × 2 s = ~64 min). The fast trio (pyzbar, OpenCV BarcodeDetector,
zxingcpp) already covers the same 1D barcode strip.

Output:
  reports/decode_trace.json      — per-tag details + aggregate stats
  reports/roi_quality_breakdown.json — failure category counts
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_ROOT   = Path("Данные")
LABELED     = ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]
OFFSETS_MS  = [0, 100, -100, 200, -200, 300, -300]

# Thresholds for ROI quality categorisation
_SMALL_PX   = 80   # min(h, w) of full crop
_BLUR_VAR   = 30   # Laplacian variance on barcode ROI
_LOW_CONT   = 20   # gray std on barcode ROI
_GLARE_FRAC = 0.20 # fraction of pixels > 240

VENV_PYTHON = "/home/nkolesnikov/.cache/pypoetry/virtualenvs/shelf-pIEl3lYF-py3.10/bin/python3"


# ── decoders (fast only) ─────────────────────────────────────────────────────

def _decode_pyzbar_raw(image: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar
        return [d.data.decode("utf-8", errors="ignore")
                for d in pyzbar.decode(image)
                if d.type in ("QRCODE", "EAN13", "EAN8", "CODE128", "CODE39")]
    except Exception:
        return []


def _decode_opencv_barcode_raw(image: np.ndarray) -> list[str]:
    try:
        det = cv2.barcode.BarcodeDetector()
        retval, decoded_info, _, _ = det.detectAndDecodeWithType(image)
        return [d for d in (decoded_info or []) if d] if retval else []
    except Exception:
        return []


def _decode_zxingcpp_raw(image: np.ndarray) -> list[str]:
    try:
        import zxingcpp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return [r.text for r in zxingcpp.read_barcodes(gray) if r.valid and r.text]
    except Exception:
        return []


_DIGITS_RE = re.compile(r"\d{8,}")


def _first_long_digits(s: str) -> str:
    m = _DIGITS_RE.search(re.sub(r"\D", "", s))
    return m.group() if m else ""


def _try_all_fast(image: np.ndarray) -> dict[str, str]:
    """Run each fast decoder, return first non-empty raw string per decoder."""
    out: dict[str, str] = {}
    for dec_name, fn in [("pyzbar", _decode_pyzbar_raw),
                         ("opencv_bc", _decode_opencv_barcode_raw),
                         ("zxingcpp", _decode_zxingcpp_raw)]:
        raws = fn(image)
        out[dec_name] = raws[0] if raws else ""
    return out


# ── quality metrics ──────────────────────────────────────────────────────────

def _sharpness(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def _contrast(gray: np.ndarray) -> float:
    return float(gray.std()) if gray.size > 0 else 0.0


def _glare_frac(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    return float((bgr.max(axis=2) > 240).mean())


def _quality_of(crop: np.ndarray) -> dict:
    """Compute quality metrics on a crop (BGR)."""
    h, w = crop.shape[:2]
    # Barcode ROI: bottom 45% of the crop
    bc_roi = crop[int(h * 0.55):, :] if h > 20 else crop
    gray_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray_bc   = cv2.cvtColor(bc_roi, cv2.COLOR_BGR2GRAY) if bc_roi.ndim == 3 else bc_roi
    return {
        "crop_h": h,
        "crop_w": w,
        "sharpness_full": round(_sharpness(gray_full), 1),
        "sharpness_bc_roi": round(_sharpness(gray_bc), 1),
        "contrast_bc_roi": round(_contrast(gray_bc), 1),
        "glare_frac": round(_glare_frac(crop), 3),
    }


def _failure_category(q: dict) -> str:
    if min(q["crop_h"], q["crop_w"]) < _SMALL_PX:
        return "too_small"
    if q["sharpness_bc_roi"] < _BLUR_VAR:
        return "too_blurry"
    if q["contrast_bc_roi"] < _LOW_CONT:
        return "low_contrast"
    if q["glare_frac"] > _GLARE_FRAC:
        return "glare"
    return "none_of_above"


# ── video helpers ─────────────────────────────────────────────────────────────

def _read_frame_at(cap: cv2.VideoCapture, ts_ms: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
    ret, frame = cap.read()
    return frame if ret else None


def _extract_crop(frame: np.ndarray, row: pd.Series) -> np.ndarray | None:
    h_f, w_f = frame.shape[:2]
    def _fc(v: object) -> float:
        return float(str(v or "0").replace(",", "."))
    x1, y1 = int(_fc(row.get("x_min", 0))), int(_fc(row.get("y_min", 0)))
    x2, y2 = int(_fc(row.get("x_max", 0))), int(_fc(row.get("y_max", 0)))
    mx = max(5, int((x2 - x1) * 0.10))
    my = max(5, int((y2 - y1) * 0.10))
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w_f, x2 + mx); y2 = min(h_f, y2 + my)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _norm_bc_str(s: str) -> str:
    s = re.sub(r"\s+", "", str(s or "").strip())
    try:
        if "." in s and s.replace(".", "").isdigit():
            s = str(int(float(s)))
        if s.isdigit() and len(s) < 13:
            s = s.zfill(13)
    except (ValueError, OverflowError):
        pass
    return s


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_video(name: str) -> list[dict]:
    csv_path = DATA_ROOT / name / f"{name}.csv"
    vid_path = DATA_ROOT / name / f"{name}.mp4"
    if not csv_path.exists() or not vid_path.exists():
        logger.warning("Missing data for %s", name)
        return []

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, decimal=",")
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})

    cap = cv2.VideoCapture(str(vid_path))
    records = []

    for _, row in df.iterrows():
        gt_bc  = _norm_bc_str(row.get("barcode", ""))
        gt_ts  = float(row.get("frame_timestamp", 0) or 0)
        tag_id = f"{name}_{int(gt_ts)}ms"

        frames_tried: list[dict] = []
        best_decoded = ""
        best_sharpness = 0.0
        quality_gt: dict = {}

        for offset_ms in OFFSETS_MS:
            ts = max(0.0, gt_ts + offset_ms)
            frame = _read_frame_at(cap, ts)
            if frame is None:
                continue
            crop = _extract_crop(frame, row)
            if crop is None:
                continue

            q = _quality_of(crop)
            if offset_ms == 0:
                quality_gt = q

            dec = _try_all_fast(crop)
            any_decoded = any(v for v in dec.values())
            sharpness = q["sharpness_full"]
            if any_decoded and sharpness > best_sharpness:
                best_sharpness = sharpness
                best_decoded   = next(v for v in dec.values() if v)

            partial_digits = {k: _first_long_digits(v)
                              for k, v in dec.items() if _first_long_digits(v)}

            frames_tried.append({
                "offset_ms": offset_ms,
                "ts_ms": int(ts),
                "sharpness": round(sharpness, 1),
                "sharpness_bc_roi": round(q["sharpness_bc_roi"], 1),
                "decoded": {k: v for k, v in dec.items() if v},
                "partial_digits": partial_digits,
                "any_decoded": any_decoded,
            })

        decoded_any_frame = any(f["any_decoded"] for f in frames_tried)
        has_partial = any(bool(f["partial_digits"]) for f in frames_tried)
        fail_cat = _failure_category(quality_gt) if quality_gt and not decoded_any_frame else (
            "decoded" if decoded_any_frame else "no_data"
        )

        # Sharpness of GT frame among decoded vs undecoded population
        gt_frame_entry = next((f for f in frames_tried if f["offset_ms"] == 0), {})
        gt_sharpness = gt_frame_entry.get("sharpness", 0)
        gt_sharpness_bc = gt_frame_entry.get("sharpness_bc_roi", 0)

        records.append({
            "tag_id": tag_id,
            "video": name,
            "gt_barcode": gt_bc,
            "gt_ts_ms": int(gt_ts),
            "gt_sharpness": gt_sharpness,
            "gt_sharpness_bc_roi": gt_sharpness_bc,
            "crop_size": (quality_gt.get("crop_h", 0), quality_gt.get("crop_w", 0)),
            "decoded_any_frame": decoded_any_frame,
            "decoded_gt_frame": frames_tried[0]["any_decoded"] if frames_tried else False,
            "best_decoded": best_decoded,
            "has_partial_digits": has_partial,
            "failure_category": fail_cat,
            "n_offsets_tried": len(frames_tried),
            "n_offsets_decoded": sum(1 for f in frames_tried if f["any_decoded"]),
            "frames": frames_tried,
        })

    cap.release()
    return records


def build_summary(records: list[dict]) -> dict:
    total = len(records)
    n_dec_any = sum(1 for r in records if r["decoded_any_frame"])
    n_dec_gt  = sum(1 for r in records if r["decoded_gt_frame"])
    n_partial = sum(1 for r in records if r["has_partial_digits"] and not r["decoded_any_frame"])

    fail_cats: dict[str, int] = defaultdict(int)
    for r in records:
        fail_cats[r["failure_category"]] += 1

    # Sharpness distributions: decoded vs undecoded (GT frame)
    sharp_dec    = [r["gt_sharpness"] for r in records if r["decoded_gt_frame"]]
    sharp_undec  = [r["gt_sharpness"] for r in records if not r["decoded_gt_frame"]]
    sharp_bc_dec  = [r["gt_sharpness_bc_roi"] for r in records if r["decoded_gt_frame"]]
    sharp_bc_und  = [r["gt_sharpness_bc_roi"] for r in records if not r["decoded_gt_frame"]]

    def _pct(lst: list, p: int) -> float:
        import statistics
        if not lst:
            return 0.0
        sorted_lst = sorted(lst)
        idx = max(0, min(len(sorted_lst) - 1, int(len(sorted_lst) * p / 100)))
        return round(sorted_lst[idx], 1)

    def _mean(lst: list) -> float:
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    per_video: dict[str, dict] = {}
    for vid in LABELED:
        vr = [r for r in records if r["video"] == vid]
        if not vr:
            continue
        per_video[vid] = {
            "n": len(vr),
            "decoded_any_frame": sum(1 for r in vr if r["decoded_any_frame"]),
            "decoded_gt_frame_only": sum(1 for r in vr if r["decoded_gt_frame"]),
            "multiframe_extra": sum(1 for r in vr
                                    if r["decoded_any_frame"] and not r["decoded_gt_frame"]),
            "partial_only": sum(1 for r in vr
                                if r["has_partial_digits"] and not r["decoded_any_frame"]),
            "failure_categories": dict(
                defaultdict(int, ((r["failure_category"], 0) for r in vr))
            ),
        }
        for r in vr:
            per_video[vid]["failure_categories"][r["failure_category"]] = \
                per_video[vid]["failure_categories"].get(r["failure_category"], 0) + 1

    return {
        "total": total,
        "decoded_any_frame": n_dec_any,
        "decoded_gt_frame": n_dec_gt,
        "multiframe_extra": n_dec_any - n_dec_gt,
        "partial_only_undecoded": n_partial,
        "decode_rate_gt_frame": round(n_dec_gt / max(1, total), 4),
        "decode_rate_any_frame": round(n_dec_any / max(1, total), 4),
        "failure_categories": dict(fail_cats),
        "sharpness": {
            "decoded_mean": _mean(sharp_dec),
            "undecoded_mean": _mean(sharp_undec),
            "decoded_p50": _pct(sharp_dec, 50),
            "undecoded_p50": _pct(sharp_undec, 50),
            "decoded_p25": _pct(sharp_dec, 25),
            "undecoded_p75": _pct(sharp_undec, 75),
            "overlap_threshold": _pct(sharp_dec, 25),
        },
        "sharpness_bc_roi": {
            "decoded_mean": _mean(sharp_bc_dec),
            "undecoded_mean": _mean(sharp_bc_und),
        },
        "per_video": per_video,
    }


def print_report(summary: dict) -> None:
    s = summary
    print(f"\n{'='*65}")
    print(f"DECODE TRACE DIAGNOSTIC  ({s['total']} GT tags)")
    print(f"{'='*65}")
    print(f"  Decoded at GT frame:          {s['decoded_gt_frame']:3d}/{s['total']}  "
          f"({s['decode_rate_gt_frame']*100:.1f}%)")
    print(f"  Decoded any frame (±300ms):   {s['decoded_any_frame']:3d}/{s['total']}  "
          f"({s['decode_rate_any_frame']*100:.1f}%)")
    print(f"  Multi-frame EXTRA decodes:    {s['multiframe_extra']:3d}  "
          f"(tags decoded ONLY via offset frames)")
    print(f"  Partial-only (8+ digits, no EAN-13): {s['partial_only_undecoded']:3d}")

    print(f"\n  Failure categories (undecoded tags):")
    total_undec = s["total"] - s["decoded_any_frame"]
    for cat, n in sorted(s["failure_categories"].items(), key=lambda x: -x[1]):
        if cat == "decoded":
            continue
        pct = 100 * n / max(1, total_undec) if total_undec else 0
        bar = "█" * int(pct / 3)
        print(f"    {cat:<20} {n:4d}  ({pct:4.1f}%)  {bar}")

    print(f"\n  Sharpness (Laplacian var, full crop):")
    sh = s["sharpness"]
    print(f"    decoded   mean={sh['decoded_mean']:6.1f}  p50={sh['decoded_p50']:6.1f}")
    print(f"    undecoded mean={sh['undecoded_mean']:6.1f}  p50={sh['undecoded_p50']:6.1f}")
    print(f"    → decoded crops are {sh['decoded_mean']/max(1,sh['undecoded_mean']):.1f}× sharper")

    print(f"\n  Per-video breakdown:")
    hdr = f"  {'Video':<14} {'n':>4}  {'dec_gt':>6}  {'dec_any':>7}  {'mf_extra':>8}  {'partial':>7}"
    print(hdr)
    for vid, pv in s["per_video"].items():
        print(f"  {vid:<14} {pv['n']:>4}  {pv['decoded_gt_frame_only']:>6}  "
              f"{pv['decoded_any_frame']:>7}  {pv['multiframe_extra']:>8}  "
              f"{pv['partial_only']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode trace per GT tag")
    ap.add_argument("--out-trace", default="reports/decode_trace.json")
    ap.add_argument("--out-quality", default="reports/roi_quality_breakdown.json")
    ap.add_argument("--video", default=None, help="Limit to one video name")
    args = ap.parse_args()

    videos = [args.video] if args.video else LABELED
    all_records: list[dict] = []
    for vid in videos:
        print(f"  Processing {vid}...", end=" ", flush=True)
        recs = analyse_video(vid)
        print(f"{len(recs)} tags")
        all_records.extend(recs)

    summary = build_summary(all_records)
    print_report(summary)

    trace_path = Path(args.out_trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(
        {"summary": summary,
         "records": [{k: v for k, v in r.items() if k != "frames"}
                     for r in all_records]},
        ensure_ascii=False, indent=2
    ))
    print(f"\nTrace → {trace_path}")

    quality_path = Path(args.out_quality)
    quality_counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        if not r["decoded_any_frame"]:
            quality_counts[r["video"]][r["failure_category"]] += 1
            quality_counts["overall"][r["failure_category"]] += 1
    quality_path.write_text(json.dumps(
        {k: dict(v) for k, v in quality_counts.items()},
        ensure_ascii=False, indent=2
    ))
    print(f"Quality breakdown → {quality_path}")


if __name__ == "__main__":
    main()
