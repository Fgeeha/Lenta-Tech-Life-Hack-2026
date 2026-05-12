"""Оркестратор end-to-end пайплайна (заглушка)."""

import logging
from pathlib import Path

import pandas as pd

from shelf.schema import OUTPUT_COLUMNS

logger = logging.getLogger(__name__)


def run(video_path: str | Path) -> pd.DataFrame:
    """Обработать видео и вернуть DataFrame по схеме OUTPUT_COLUMNS.

    На этапе 0 возвращает пустой DataFrame с правильными столбцами.
    """
    video_path = Path(video_path)
    logger.info("Pipeline stub: %s", video_path.name)
    return pd.DataFrame(columns=OUTPUT_COLUMNS)
