"""SR-on-ROI barcode decode test (Task C).

Applies RealESRGAN 4× upscaling ONLY to the barcode ROI (bottom 45% of
price-tag crop), then runs the full fast-decoder cascade on the upscaled
image. Reports how many previously-undecoded tags become decodable.

Usage (dry-run, no pipeline changes):
  python scripts/sr_barcode_only.py

Requires:
  - reports/decode_trace.json  (from diag_decode_per_tag.py)
  - models/RealESRGAN_x4plus.pth
  - realesrgan + basicsr packages (already in venv)

Do NOT run this while the multiframe ceiling eval is active:
  ps aux | grep eval_ceiling   ← check first

Decision gate: if sr_decode_rate > 0.30 (out of undecoded pool) → wire
into the decode pipeline in src/shelf/qr/decoder.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_ROOT  = Path("Данные")
LABELED    = ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]
_DEFAULT_N = 50   # max undecoded tags to test (for speed)


# ── SR upsampler ─────────────────────────────────────────────────────────────

def _load_sr() -> bool:
    """Warm up SR model. Returns True if model loaded successfully."""
    try:
        import os
        os.environ["SHELF_SR_ENABLED"] = "1"
        from shelf.ocr.sr import upscale_crop
        probe = np.zeros((16, 16, 3), dtype=np.uint8)
        result = upscale_crop(probe)
        return result.shape[0] > 16
    except Exception as exc:
        logger.error("SR warm-up error: %s", exc)
        return False


def _upscale_roi(roi: np.ndarray, _unused=None) -> np.ndarray | None:
    """Run RealESRGAN on a small ROI via the public upscale_crop() function."""
    try:
        from shelf.ocr.sr import upscale_crop
        result = upscale_crop(roi)
        return result if result is not None and result.size > 0 else None
    except Exception as exc:
        logger.debug("SR enhance failed: %s", exc)
        return None


# ── fast decoders (same as diag_decode_per_tag.py) ───────────────────────────

def _decode_pyzbar(image: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar
        return [d.data.decode("utf-8", errors="ignore")
                for d in pyzbar.decode(image)
                if d.type in ("QRCODE", "EAN13", "EAN8", "CODE128")]
    except Exception:
        return []


def _decode_opencv_bc(image: np.ndarray) -> list[str]:
    try:
        det = cv2.barcode.BarcodeDetector()
        retval, decoded_info, _, _ = det.detectAndDecodeWithType(image)
        return [d for d in (decoded_info or []) if d] if retval else []
    except Exception:
        return []


def _decode_zxingcpp(image: np.ndarray) -> list[str]:
    try:
        import zxingcpp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return [r.text for r in zxingcpp.read_barcodes(gray) if r.valid and r.text]
    except Exception:
        return []


_DIGITS_RE = re.compile(r"\d{8,}")


def _try_decode(image: np.ndarray) -> str:
    """Return first raw decode string from any fast decoder."""
    for fn in [_decode_pyzbar, _decode_opencv_bc, _decode_zxingcpp]:
        raws = fn(image)
        if raws:
            return raws[0]
    return ""


def _extract_barcode_roi(crop: np.ndarray) -> np.ndarray:
    """Extract the bottom 45% of a price-tag crop (where the barcode lives)."""
    h = crop.shape[0]
    roi = crop[int(h * 0.55):, :]
    return roi if roi.size > 0 else crop


# ── video helpers ─────────────────────────────────────────────────────────────

def _norm_bc(val: object) -> str:
    s = re.sub(r"\s+", "", str(val or "").strip())
    try:
        if "." in s and s.replace(".", "").isdigit():
            s = str(int(float(s)))
        if s.isdigit() and len(s) < 13:
            s = s.zfill(13)
    except (ValueError, OverflowError):
        pass
    return s


def _extract_crop(frame: np.ndarray, row: pd.Series) -> np.ndarray | None:
    h_f, w_f = frame.shape[:2]
    try:
        x1, y1 = int(float(row.get("x_min", 0))), int(float(row.get("y_min", 0)))
        x2, y2 = int(float(row.get("x_max", 0))), int(float(row.get("y_max", 0)))
    except (ValueError, TypeError):
        return None
    mx = max(5, int((x2 - x1) * 0.10))
    my = max(5, int((y2 - y1) * 0.10))
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w_f, x2 + mx); y2 = min(h_f, y2 + my)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _load_undecoded_from_trace(trace_path: Path, max_n: int) -> list[dict]:
    """Load undecoded tag specs from decode_trace.json."""
    if not trace_path.exists():
        logger.warning("decode_trace.json not found at %s", trace_path)
        return []
    data = json.loads(trace_path.read_text())
    records = data.get("records", [])
    undecoded = [r for r in records if not r.get("decoded_any_frame")]
    # Prefer videos with more undecoded tags to sample broadly
    by_video: dict[str, list] = {}
    for r in undecoded:
        by_video.setdefault(r["video"], []).append(r)
    selected: list[dict] = []
    per_video = max(1, max_n // max(1, len(by_video)))
    for vid_recs in by_video.values():
        selected.extend(vid_recs[:per_video])
    return selected[:max_n]


def _fallback_undecoded_from_gt(max_n: int) -> list[dict]:
    """Build undecoded list directly from GT CSVs (if no trace file exists)."""
    records = []
    for name in LABELED:
        csv_p = DATA_ROOT / name / f"{name}.csv"
        if not csv_p.exists():
            continue
        df = pd.read_csv(csv_p, dtype=str, keep_default_na=False, decimal=",")
        for _, row in df.iterrows():
            records.append({
                "video": name,
                "gt_barcode": _norm_bc(row.get("barcode", "")),
                "gt_ts_ms": int(float(row.get("frame_timestamp", 0) or 0)),
                "x_min": row.get("x_min", "0"),
                "x_max": row.get("x_max", "0"),
                "y_min": row.get("y_min", "0"),
                "y_max": row.get("y_max", "0"),
            })
    return records[:max_n]


# ── main test ─────────────────────────────────────────────────────────────────

def run_sr_test(
    undecoded: list[dict],
    upsampler,
    *,
    verbose: bool = False,
) -> dict:
    results = []
    caps: dict[str, cv2.VideoCapture] = {}
    t0 = time.time()

    for i, tag in enumerate(undecoded):
        vid    = tag["video"]
        ts_ms  = float(tag.get("gt_ts_ms", 0))
        gt_bc  = tag.get("gt_barcode", "")

        # Open video (lazy, cached)
        if vid not in caps:
            vid_path = DATA_ROOT / vid / f"{vid}.mp4"
            if not vid_path.exists():
                continue
            caps[vid] = cv2.VideoCapture(str(vid_path))

        cap = caps[vid]
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()
        if not ret:
            continue

        # Build a minimal Series-like dict for _extract_crop
        row = pd.Series(tag)
        crop = _extract_crop(frame, row)
        if crop is None:
            continue

        bc_roi = _extract_barcode_roi(crop)

        # Baseline: try fast decoders on original ROI (should fail — tag is undecoded)
        baseline_raw = _try_decode(bc_roi)
        baseline_ok  = bool(baseline_raw)

        # SR on barcode ROI
        t_sr = time.time()
        sr_roi = _upscale_roi(bc_roi, upsampler)
        sr_ms  = round((time.time() - t_sr) * 1000, 1)

        sr_raw = _try_decode(sr_roi) if sr_roi is not None else ""
        sr_ok  = bool(sr_raw)

        # Also try SR on full crop
        sr_full = _upscale_roi(crop, upsampler) if sr_roi is not None else None
        sr_full_raw = _try_decode(sr_full) if sr_full is not None else ""
        sr_full_ok  = bool(sr_full_raw)

        r = {
            "tag_id": f"{vid}_{int(ts_ms)}ms",
            "video": vid,
            "gt_barcode": gt_bc,
            "baseline_ok": baseline_ok,
            "sr_roi_ok": sr_ok,
            "sr_full_ok": sr_full_ok,
            "sr_roi_decoded": sr_raw[:20] if sr_ok else "",
            "sr_full_decoded": sr_full_raw[:20] if sr_full_ok else "",
            "sr_ms": sr_ms,
            "roi_size": (bc_roi.shape[0], bc_roi.shape[1]),
        }
        results.append(r)
        if verbose:
            icon = "✓" if sr_ok else "✗"
            print(f"  [{i+1:3d}/{len(undecoded)}] {vid} t={ts_ms:.0f}ms  "
                  f"roi={bc_roi.shape[1]}×{bc_roi.shape[0]}  "
                  f"sr_roi={icon}  sr_full={'✓' if sr_full_ok else '✗'}  "
                  f"{sr_ms}ms")

    for cap in caps.values():
        cap.release()

    total_tested = len(results)
    n_sr_roi  = sum(1 for r in results if r["sr_roi_ok"])
    n_sr_full = sum(1 for r in results if r["sr_full_ok"])
    n_either  = sum(1 for r in results if r["sr_roi_ok"] or r["sr_full_ok"])
    elapsed   = round(time.time() - t0, 1)

    return {
        "total_tested": total_tested,
        "sr_roi_decoded": n_sr_roi,
        "sr_full_decoded": n_sr_full,
        "sr_either_decoded": n_either,
        "sr_roi_rate": round(n_sr_roi / max(1, total_tested), 3),
        "sr_full_rate": round(n_sr_full / max(1, total_tested), 3),
        "sr_either_rate": round(n_either / max(1, total_tested), 3),
        "avg_sr_ms_per_tag": round(elapsed * 1000 / max(1, total_tested), 0),
        "total_elapsed_s": elapsed,
        "decision": (
            "WIRE_INTO_PIPELINE" if n_either / max(1, total_tested) >= 0.30
            else "MARGINAL"      if n_either / max(1, total_tested) >= 0.15
            else "NOT_USEFUL"
        ),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SR-on-ROI barcode decode test")
    ap.add_argument("--trace", default="reports/decode_trace.json")
    ap.add_argument("--n", type=int, default=_DEFAULT_N, help="Max undecoded tags to test")
    ap.add_argument("--out", default="reports/sr_barcode_test.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("Loading SR model...")
    import os
    os.environ["SHELF_SR_ENABLED"] = "1"
    if not _load_sr():
        print("ERROR: SR model not available. Run: ls models/RealESRGAN_x4plus.pth")
        sys.exit(1)
    print("SR model loaded.")

    trace_path = Path(args.trace)
    undecoded = _load_undecoded_from_trace(trace_path, args.n)
    if not undecoded:
        print(f"No decode_trace.json found at {trace_path}. "
              "Run diag_decode_per_tag.py first. Falling back to GT CSVs.")
        undecoded = _fallback_undecoded_from_gt(args.n)

    print(f"Testing SR on {len(undecoded)} undecoded tags from "
          f"{len(set(r['video'] for r in undecoded))} videos...")

    result = run_sr_test(undecoded, None, verbose=args.verbose)

    print(f"\n{'='*55}")
    print(f"SR-ON-BARCODE-ROI RESULTS  ({result['total_tested']} tags tested)")
    print(f"{'='*55}")
    print(f"  SR on barcode ROI only:  {result['sr_roi_decoded']:3d}/{result['total_tested']}"
          f"  ({result['sr_roi_rate']*100:.1f}%)")
    print(f"  SR on full crop:         {result['sr_full_decoded']:3d}/{result['total_tested']}"
          f"  ({result['sr_full_rate']*100:.1f}%)")
    print(f"  Either method decoded:   {result['sr_either_decoded']:3d}/{result['total_tested']}"
          f"  ({result['sr_either_rate']*100:.1f}%)")
    print(f"  Avg SR time per tag:     {result['avg_sr_ms_per_tag']:.0f}ms")
    print(f"\n  Decision: {result['decision']}")
    if result["decision"] == "WIRE_INTO_PIPELINE":
        print("  → SR boosts decode rate ≥30% on undecoded pool.")
        print("    Wire into decoder.py before the fast-decoder cascade.")
    elif result["decision"] == "MARGINAL":
        print("  → Marginal gain (15-30%). Consider only for 43_15-type videos.")
    else:
        print("  → SR does not help enough. Image quality is the hard floor.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
