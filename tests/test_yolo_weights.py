"""Tests for local-only YOLO weight discovery."""

from shelf.detect.yolo_sahi import _candidate_weight_paths, _ensure_weights


def test_yolo_weights_no_download_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SHELF_ALLOW_MODEL_DOWNLOAD", raising=False)
    monkeypatch.delenv("SHELF_YOLO_WEIGHTS", raising=False)
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.pt"
    assert _ensure_weights(str(missing)) is None


def test_yolo_weights_env_priority(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"local")
    monkeypatch.setenv("SHELF_YOLO_WEIGHTS", str(weights))
    candidates = _candidate_weight_paths("models/pricetag_tiled_yolov8n.pt")
    assert candidates[0] == weights
    assert _ensure_weights("models/pricetag_tiled_yolov8n.pt") == str(weights)
