"""Tests for OCR engine configuration without loading heavy OCR models."""

from shelf.ocr.engine import OCREngine, _auto_order, normalize_ocr_engine_name


def test_normalize_ocr_engine_aliases(monkeypatch):
    monkeypatch.delenv("SHELF_OCR_ENGINE", raising=False)
    assert normalize_ocr_engine_name("ppocr5") == "paddle_v5"
    assert normalize_ocr_engine_name("paddle") == "paddle_v4"
    assert normalize_ocr_engine_name("disabled") == "none"
    assert normalize_ocr_engine_name("unknown") == "auto"


def test_auto_order_env(monkeypatch):
    monkeypatch.setenv("SHELF_OCR_AUTO_ORDER", "paddle_v5,easyocr")
    assert _auto_order() == ["paddle_v5", "easyocr"]


def test_paddle_v5_constructor_attempts_include_mobile_names():
    engine = OCREngine(lang="ru", engine="paddle_v5")
    attempts = engine._paddle_constructor_attempts("paddle_v5", {})
    assert attempts[0]["ocr_version"] == "PP-OCRv5"
    assert attempts[0]["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert attempts[0]["text_recognition_model_name"] == "PP-OCRv5_mobile_rec"


def test_force_easyocr_overrides_env(monkeypatch):
    monkeypatch.setenv("SHELF_OCR_ENGINE", "paddle_v5")
    engine = OCREngine(lang="ru", force_easyocr=True)
    assert engine.engine_name == "easyocr"
