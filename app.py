"""Точка входа для Gradio / HF Spaces."""

import logging

from shelf.ui.gradio_app import build_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
