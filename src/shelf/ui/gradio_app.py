"""Gradio UI для запуска пайплайна."""

import logging
import tempfile

import gradio as gr
import pandas as pd

from shelf import pipeline

logger = logging.getLogger(__name__)


def process_video(video_path: str | None) -> tuple[str, pd.DataFrame]:
    if video_path is None:
        return "Видео не загружено", pd.DataFrame()
    try:
        df = pipeline.run(video_path)
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(out.name, index=False)
        return out.name, df.head(20)
    except Exception as exc:
        logger.exception("Pipeline error")
        return str(exc), pd.DataFrame()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Полка под контролем") as app:
        gr.Markdown("## Полка под контролем\nЗагрузи видео — получи CSV с ценниками.")
        with gr.Row():
            video_input = gr.Video(label="Видео (.mp4)")
        run_btn = gr.Button("Запустить", variant="primary")
        with gr.Row():
            csv_output = gr.File(label="Скачать CSV")
            table_output = gr.Dataframe(label="Превью (первые 20 строк)")
        run_btn.click(fn=process_video, inputs=[video_input], outputs=[csv_output, table_output])
    return app
