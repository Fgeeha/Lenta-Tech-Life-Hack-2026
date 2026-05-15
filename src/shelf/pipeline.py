"""End-to-end orchestrator: video -> unique price-tag rows -> CSV."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from shelf.detect.detector import make_detector
from shelf.detect.tracker import TrackCandidate, Tracker
from shelf.io.video import sample_frames
from shelf.io.writer import prepare_output_dataframe, write_csv
from shelf.ocr.engine import OCREngine
from shelf.ocr.parser import parse_ocr_result
from shelf.ocr.preprocess import ocr_variants
from shelf.ocr.template import classify_color
from shelf.postproc.catalog import apply_catalog, load_catalog_from_env
from shelf.postproc.dedup import deduplicate_tags, tag_completeness
from shelf.postproc.merge import merge
from shelf.postproc.pass80 import optimize_tags
from shelf.postproc.voting import merge_candidate_tags
from shelf.qr.decoder import decode_barcode, decode_qr
from shelf.schema import OUTPUT_COLUMNS, PriceTag

logger = logging.getLogger(__name__)

_CROP_MARGIN = 24


def _append_debug_row(
    path: Path, fieldnames: list[str], row: dict[str, object]
) -> None:
    """Append a structured debug row without polluting normal runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _limit_tracks_for_ocr(best_tracks: dict[int, object]) -> dict[int, object]:
    """Optionally keep only top-scored tracks for fast smoke/debug runs.

    Production default is unlimited.  Set ``SHELF_MAX_TRACKS`` only for local
    smoke tests or demos when OCR/QR on all detected crops would be too slow.
    """
    max_tracks_raw = os.getenv("SHELF_MAX_TRACKS", "").strip()
    if not max_tracks_raw:
        return best_tracks
    try:
        max_tracks = int(max_tracks_raw)
    except ValueError:
        logger.warning("Invalid SHELF_MAX_TRACKS=%r; ignoring", max_tracks_raw)
        return best_tracks
    if max_tracks <= 0 or len(best_tracks) <= max_tracks:
        return best_tracks
    return dict(
        sorted(
            best_tracks.items(),
            key=lambda item: getattr(item[1], "best_score", 0.0),
            reverse=True,
        )[:max_tracks]
    )


def _extract_tag(
    crop_raw,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    filename: str,
    timestamp_ms: float,
    ocr_engine: OCREngine,
    ocr_engine_ru: OCREngine | None = None,
    max_ocr_variants: int = 2,
    debug_dir: str | Path | None = None,
    track_id: int | None = None,
) -> PriceTag:
    """Process one crop: QR + color + OCR + parser + merge."""
    base = PriceTag(
        filename=filename,
        frame_timestamp=timestamp_ms,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
    if crop_raw is None or crop_raw.size == 0:
        return base

    qr_fields = decode_qr(
        crop_raw,
        debug_dir=debug_dir,
        track_id=track_id,
        timestamp_ms=timestamp_ms,
    )
    linear_barcode = decode_barcode(
        crop_raw,
        debug_dir=debug_dir,
        track_id=track_id,
        timestamp_ms=timestamp_ms,
    )
    if linear_barcode and not qr_fields.get("barcode"):
        qr_fields["barcode"] = linear_barcode
    color = classify_color(crop_raw)

    best_tag: PriceTag | None = None
    variants = ocr_variants(crop_raw) or [crop_raw]
    for proc in variants[: max(1, max_ocr_variants)]:
        try:
            ocr_lines = ocr_engine.run(proc)
        except Exception as exc:
            logger.warning("OCR failed on crop: %s", exc)
            ocr_lines = []

        ocr_tag = parse_ocr_result(
            ocr_lines,
            crop=proc,
            crop_raw=crop_raw,
            filename=filename,
            frame_timestamp=timestamp_ms,
            bbox=(x_min, y_min, x_max, y_max),
            color=color,
            ocr_ru=ocr_engine_ru,
        )
        tag = merge(ocr_tag, qr_fields)
        if best_tag is None or tag_completeness(tag) > tag_completeness(
            best_tag
        ):
            best_tag = tag
        # Fast exit: enough high-value fields are already recognized.
        if tag.price_card and (tag.barcode or tag.qr_code_barcode != "нет"):
            break

    return best_tag or merge(base, qr_fields)


def _candidate_to_tag(
    candidate: TrackCandidate,
    filename: str,
    ocr_engine: OCREngine,
    ocr_engine_ru: OCREngine | None,
    max_ocr_variants: int,
    debug_dir: str | Path | None = None,
    track_id: int | None = None,
) -> PriceTag:
    d = candidate.det
    return _extract_tag(
        crop_raw=candidate.crop,
        x_min=d.x_min,
        y_min=d.y_min,
        x_max=d.x_max,
        y_max=d.y_max,
        filename=filename,
        timestamp_ms=candidate.timestamp_ms,
        ocr_engine=ocr_engine,
        ocr_engine_ru=ocr_engine_ru,
        max_ocr_variants=max_ocr_variants,
        debug_dir=debug_dir,
        track_id=track_id,
    )


def run(
    video_path: str | Path,
    interval_ms: int = 250,
    adaptive: bool = True,
    min_hits: int = 2,
    output_csv: str | Path | None = None,
    detector_name: str = "hybrid",
    max_duration_sec: float | None = None,
    ocr_top_k: int = 2,
    max_ocr_variants: int = 2,
    ocr_engine_name: str | None = None,
    debug_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Process a video and return a DataFrame with the required CSV schema.

    ``frame_timestamp`` is stored in milliseconds from the beginning of the video.
    """
    video_path = Path(video_path)
    filename = video_path.name

    detector = make_detector(detector_name)
    tracker = Tracker(
        min_hits=min_hits,
        crop_margin=_CROP_MARGIN,
        max_candidates=max(1, ocr_top_k),
    )
    ocr_engine = OCREngine(engine=ocr_engine_name)
    ocr_engine_ru = OCREngine(lang="ru", force_easyocr=True)

    logger.info("Запуск пайплайна: %s detector=%s", filename, detector_name)
    frame_count = 0
    max_ts_ms = max_duration_sec * 1000.0 if max_duration_sec else None

    def _progress(value: float, text: str) -> None:
        if progress_callback:
            progress_callback(value, text)

    for ts_ms, frame in sample_frames(
        video_path,
        interval_ms=interval_ms,
        adaptive=adaptive,
        max_timestamp_ms=max_ts_ms,
        progress_callback=lambda v, t: _progress(v, t),
    ):
        if max_ts_ms is not None and ts_ms > max_ts_ms:
            logger.info(
                "Достигнут лимит %.0fс — остановка сэмплирования",
                max_duration_sec,
            )
            break
        dets = detector.detect(frame)
        tracker.update(dets, frame, ts_ms)
        frame_count += 1
        if frame_count % 25 == 0:
            _progress(
                0.35 + min(0.35, frame_count / 300.0 * 0.35),
                f"Кадры: {frame_count}, треки: {len(tracker._states)}",
            )
            logger.info(
                "Кадры: %d  треки: %d", frame_count, len(tracker._states)
            )

    best_tracks = tracker.get_best_crops(min_hits=min_hits)
    before_limit = len(best_tracks)
    best_tracks = _limit_tracks_for_ocr(best_tracks)
    if len(best_tracks) != before_limit:
        logger.info(
            "Ограничили OCR треки: %d -> %d", before_limit, len(best_tracks)
        )
    logger.info(
        "Стабильных треков (%d+ кадров): %d", min_hits, len(best_tracks)
    )
    _progress(0.75, f"OCR треков: {len(best_tracks)}")

    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    tags: list[PriceTag] = []
    for idx, (tid, state) in enumerate(best_tracks.items(), start=1):
        candidates = state.candidates[: max(1, ocr_top_k)]
        if not candidates and state.best_frame is not None:
            candidates = [
                TrackCandidate(
                    state.best_det,
                    state.best_frame,
                    state.best_ts,
                    state.best_score,
                    0.0,
                )
            ]
        if not candidates:
            continue

        candidate_tags: list[PriceTag] = []
        for ci, cand in enumerate(candidates):
            if debug_path is not None:
                cv2.imwrite(
                    str(
                        debug_path
                        / f"track_{tid}_cand_{ci}_{int(cand.timestamp_ms)}ms.jpg"
                    ),
                    cand.crop,
                )
            tag = _candidate_to_tag(
                cand,
                filename,
                ocr_engine,
                ocr_engine_ru,
                max_ocr_variants=max_ocr_variants,
                debug_dir=debug_path,
                track_id=int(tid),
            )
            if debug_path is not None:
                _append_debug_row(
                    debug_path / "product_name_candidates.csv",
                    [
                        "track_id",
                        "candidate_idx",
                        "timestamp",
                        "score",
                        "product_name",
                    ],
                    {
                        "track_id": tid,
                        "candidate_idx": ci,
                        "timestamp": int(cand.timestamp_ms),
                        "score": f"{cand.score:.4f}",
                        "product_name": tag.product_name,
                    },
                )
                _append_debug_row(
                    debug_path / "price_candidates.csv",
                    [
                        "track_id",
                        "candidate_idx",
                        "timestamp",
                        "score",
                        "price_default",
                        "price_card",
                        "price_discount",
                        "discount_amount",
                        "price1_qr",
                        "price4_qr",
                    ],
                    {
                        "track_id": tid,
                        "candidate_idx": ci,
                        "timestamp": int(cand.timestamp_ms),
                        "score": f"{cand.score:.4f}",
                        "price_default": tag.price_default,
                        "price_card": tag.price_card,
                        "price_discount": tag.price_discount,
                        "discount_amount": tag.discount_amount,
                        "price1_qr": tag.price1_qr,
                        "price4_qr": tag.price4_qr,
                    },
                )
            candidate_tags.append(tag)
        best = merge_candidate_tags(
            candidate_tags, candidate_scores=[c.score for c in candidates]
        )
        tags.append(best)
        if progress_callback and len(best_tracks) > 0:
            _progress(
                0.75 + 0.20 * idx / len(best_tracks),
                f"OCR: {idx}/{len(best_tracks)}",
            )

    # Trackers can split one physical tag; merge duplicate rows conservatively.
    tags = deduplicate_tags(tags)
    # Optional local catalog lookup from data/catalog.csv or SHELF_CATALOG_PATH.
    catalog = load_catalog_from_env()
    tags = apply_catalog(tags, catalog)
    tags, pass80_report = optimize_tags(
        tags, catalog=catalog, debug_dir=debug_path
    )
    if pass80_report.changes:
        logger.info(
            "Pass80 optimizer: %d field updates, proxy_crossed=%d",
            len(pass80_report.changes),
            pass80_report.proxy_crossed_80,
        )
    df = prepare_output_dataframe(
        pd.DataFrame([t.to_dict() for t in tags], columns=OUTPUT_COLUMNS)
    )

    if output_csv is not None:
        write_csv(df, output_csv)
        logger.info("CSV сохранён: %s", output_csv)

    _progress(1.0, f"Готово: {len(df)} уникальных ценников")
    logger.info("Готово: %d уникальных ценников", len(df))
    return df
