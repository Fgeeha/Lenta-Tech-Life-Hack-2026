"""Gradio UI for video -> CSV processing."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from shelf import pipeline
from shelf.io.writer import write_csv

logger = logging.getLogger(__name__)

_PREVIEW_COLS = [
    "product_name",
    "price_card",
    "price_default",
    "discount_amount",
    "barcode",
    "qr_code_barcode",
    "color",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]

_MAX_DURATION_SEC = (
    180  # CPU-friendly safety limit; set 0 in UI to disable locally.
)


def _video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


def process_video(
    video_path: str | None,
    interval_ms: int,
    min_hits: int,
    adaptive: bool,
    detector_name: str,
    ocr_top_k: int,
    max_duration_sec: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str | None, pd.DataFrame, str]:
    """Process an uploaded video and return (csv_path, preview_df, status)."""
    if video_path is None:
        return None, pd.DataFrame(), "Видео не загружено"

    dur = _video_duration(video_path)
    limit = (
        float(max_duration_sec)
        if max_duration_sec and max_duration_sec > 0
        else None
    )
    warn = ""
    if limit and dur > limit:
        warn = f"Видео {dur:.0f}с > лимита {limit:.0f}с — обрабатываем первые {limit:.0f}с.\n"
        logger.warning("Video %.0fs > limit %.0fs, truncating", dur, limit)

    def _progress(frac: float, desc: str) -> None:
        try:
            progress(float(max(0.0, min(1.0, frac))), desc=desc)
        except Exception:
            pass

    try:
        logger.info(
            "Обработка %s (детектор: %s)", Path(video_path).name, detector_name
        )
        _progress(0.02, "Старт пайплайна")
        df = pipeline.run(
            video_path,
            interval_ms=int(interval_ms),
            adaptive=bool(adaptive),
            min_hits=int(min_hits),
            detector_name=detector_name,
            max_duration_sec=limit,
            ocr_top_k=int(ocr_top_k),
            progress_callback=_progress,
        )

        if df.empty:
            return (
                None,
                pd.DataFrame(),
                warn
                + "Ценники не найдены. Попробуйте detector=hybrid/mser, меньший interval_ms или min_hits=1.",
            )

        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="shelf_"
        )
        tmp.close()
        write_csv(df, tmp.name)

        preview_cols = [c for c in _PREVIEW_COLS if c in df.columns]
        preview = df[preview_cols].head(100)

        price_count = (
            int((df["price_card"].astype(str) != "").sum())
            if "price_card" in df
            else 0
        )
        barcode_count = (
            int((df["barcode"].astype(str) != "").sum())
            if "barcode" in df
            else 0
        )
        qr_count = (
            int((~df["qr_code_barcode"].astype(str).isin(["", "нет"])).sum())
            if "qr_code_barcode" in df
            else 0
        )
        status_msg = (
            warn + f"Найдено уникальных ценников: {len(df)}\n"
            f"Строк с price_card: {price_count}\n"
            f"Строк с barcode: {barcode_count}\n"
            f"Строк с QR barcode: {qr_count}"
        )
        _progress(1.0, "Готово")
        return tmp.name, preview, status_msg

    except Exception as exc:
        logger.exception("Pipeline error")
        return None, pd.DataFrame(), f"Ошибка: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Полка под контролем", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Полка под контролем\n"
            "Загрузите видео с робота-сканера Lenta — получите CSV по схеме ТЗ: одна строка = один уникальный ценник.\n\n"
            "Рекомендуемый режим: **hybrid** detector + adaptive sampling + OCR top-2 кадра на трек."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Видео (.mp4/.mov)", height=300)

                with gr.Accordion("Параметры качества/скорости", open=True):
                    detector_radio = gr.Radio(
                        choices=[
                            "hybrid",
                            "yolo-tiled",
                            "mser",
                            "yolo-ft",
                            "yolo",
                        ],
                        value="hybrid",
                        label="Детектор",
                        info="hybrid = trained tiled YOLO если есть веса + MSER fallback",
                    )
                    interval_slider = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=300,
                        step=50,
                        label="Интервал семплирования (мс)",
                        info="Меньше = выше recall, медленнее. 250-400 мс обычно оптимально.",
                    )
                    min_hits_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="min_hits",
                        info="1 — максимум recall; 2-3 — меньше ложных ценников.",
                    )
                    ocr_top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="OCR top-K кадров на трек",
                        info="2-3 улучшает OCR на бликах/размытии, но медленнее.",
                    )
                    duration_slider = gr.Slider(
                        minimum=0,
                        maximum=600,
                        value=_MAX_DURATION_SEC,
                        step=10,
                        label="Лимит длительности, сек (0 = без лимита)",
                    )
                    adaptive_check = gr.Checkbox(
                        value=True,
                        label="Адаптивный семплинг: пропуск почти одинаковых кадров",
                    )

                run_btn = gr.Button(
                    "Запустить распознавание", variant="primary", size="lg"
                )
                status_box = gr.Textbox(
                    label="Статус", lines=5, interactive=False
                )

            with gr.Column(scale=2):
                csv_output = gr.File(label="Скачать CSV")
                table_output = gr.Dataframe(
                    label="Превью результата", wrap=True, interactive=False
                )

        run_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                interval_slider,
                min_hits_slider,
                adaptive_check,
                detector_radio,
                ocr_top_k_slider,
                duration_slider,
            ],
            outputs=[csv_output, table_output, status_box],
        )

        gr.Markdown(
            "---\n"
            "CSV всегда сохраняется в порядке колонок ТЗ: filename, product_name, price_default, price_card, "
            "price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, "
            "color, special_symbols, frame_timestamp, bbox + QR-поля."
        )

    return app
