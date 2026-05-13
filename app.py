"""Entry point for Gradio / Hugging Face Spaces."""

import logging

from shelf.ui.gradio_app import build_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
