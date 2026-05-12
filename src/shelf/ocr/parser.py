"""Маппинг OCR-боксов → поля ценника (заглушка)."""

from shelf.schema import PriceTag


def parse_ocr_result(
    ocr_lines: list[tuple[list, str, float]],
    template_color: str = "white",
    filename: str = "",
    frame_timestamp: float = 0.0,
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> PriceTag:
    """Преобразует OCR-текст в PriceTag. Заглушка: пишем всё в additional_info."""
    all_text = " | ".join(text for _, text, _ in ocr_lines)
    x_min, y_min, x_max, y_max = bbox
    return PriceTag(
        filename=filename,
        frame_timestamp=frame_timestamp,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        additional_info=all_text,
        color=template_color,
    )
