"""Gradio UI для запуска пайплайна видео → CSV."""

import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from shelf import pipeline

logger = logging.getLogger(__name__)

_PREVIEW_COLS = [
    "product_name",
    "price_card",
    "price_default",
    "discount_amount",
    "barcode",
    "color",
    "frame_timestamp",
]

_MAX_DURATION_SEC = 90  # лимит для HF Spaces CPU


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
) -> tuple[str | None, pd.DataFrame, str]:
    """Обработать загруженное видео, вернуть (csv_path, preview_df, status)."""
    if video_path is None:
        return None, pd.DataFrame(), "Видео не загружено"

    dur = _video_duration(video_path)
    warn = ""
    if dur > _MAX_DURATION_SEC:
        warn = f"Видео {dur:.0f}с > лимита {_MAX_DURATION_SEC}с — обрабатываем первые {_MAX_DURATION_SEC}с.\n"
        logger.warning("Video %.0fs > limit %ds, truncating", dur, _MAX_DURATION_SEC)

    try:
        logger.info("Обработка %s (детектор: %s)", Path(video_path).name, detector_name)
        df = pipeline.run(
            video_path,
            interval_ms=int(interval_ms),
            adaptive=bool(adaptive),
            min_hits=int(min_hits),
            detector_name=detector_name,
            max_duration_sec=_MAX_DURATION_SEC,
        )

        if df.empty:
            return None, pd.DataFrame(), warn + "Ценники не найдены"

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, prefix="shelf_")
        df.to_csv(tmp.name, index=False)

        preview_cols = [c for c in _PREVIEW_COLS if c in df.columns]
        preview = df[preview_cols].head(50)

        status_msg = (
            warn
            + f"Найдено {len(df)} уникальных ценников.\n"
            f"Строк с price_card: {(df.price_card != '').sum()}\n"
            f"Строк с barcode: {(df.barcode != '').sum()}"
        )
        return tmp.name, preview, status_msg

    except Exception as exc:
        logger.exception("Pipeline error")
        return None, pd.DataFrame(), f"Ошибка: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Полка под контролем", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Полка под контролем\n"
            "Загрузи видео с робота-сканера Lenta → получи CSV с распознанными ценниками.\n\n"
            f"> Лимит на HF Spaces CPU: первые **{_MAX_DURATION_SEC} секунд** видео."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Видео (.mp4)", height=300)

                with gr.Accordion("Параметры", open=False):
                    detector_radio = gr.Radio(
                        choices=["yolo-tiled", "yolo-ft", "mser", "yolo"],
                        value="yolo-tiled",
                        label="Детектор",
                        info="yolo-tiled = тайловый YOLO 4K (рекомендуется), mser = без обучения",
                    )
                    interval_slider = gr.Slider(
                        minimum=200,
                        maximum=2000,
                        value=500,
                        step=100,
                        label="Интервал семплирования (мс)",
                        info="500мс — баланс скорость/качество на CPU",
                    )
                    min_hits_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="min_hits (фильтр треков)",
                    )
                    adaptive_check = gr.Checkbox(value=True, label="Адаптивный семплинг (пропуск статики)")

                run_btn = gr.Button("Запустить распознавание", variant="primary", size="lg")
                status_box = gr.Textbox(label="Статус", lines=4, interactive=False)

            with gr.Column(scale=2):
                csv_output = gr.File(label="Скачать CSV")
                table_output = gr.Dataframe(
                    label="Превью (первые 50 строк)",
                    wrap=True,
                )

        run_btn.click(
            fn=process_video,
            inputs=[video_input, interval_slider, min_hits_slider, adaptive_check, detector_radio],
            outputs=[csv_output, table_output, status_box],
        )

        gr.Markdown(
            "---\n"
            "**Поля CSV:** filename, product_name, price_default, price_card, "
            "price_discount, barcode, discount_amount, id_sku, print_datetime, "
            "code, additional_info, color, special_symbols, frame_timestamp, "
            "x_min, y_min, x_max, y_max + QR-поля\n\n"
            "**metric@80% = 0.013** на 3 размеченных видео · детекция 157/157 · "
            "[GitHub](https://github.com/nkolesnikov/Lenta-Tech-Life-Hack-2026)"
        )

    return app
