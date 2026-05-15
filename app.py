"""Entry point for Gradio / Hugging Face Spaces."""

import logging
import os
import sys

# src/ layout: add to path for HF Spaces (no pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from shelf.ui.gradio_app import build_app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
