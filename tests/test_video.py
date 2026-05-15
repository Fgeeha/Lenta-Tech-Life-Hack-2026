"""Тесты frame sampler (без реального видео — smoke-тест на синтетике)."""

import numpy as np
import pytest


def _make_fake_video(
    tmp_path, n_frames: int = 30, w: int = 320, h: int = 240, fps: float = 20.0
):
    """Создать синтетическое видео для тестов."""
    import cv2

    out_path = tmp_path / "fake.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), fill_value=i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out_path


def test_sample_frames_count(tmp_path):
    from shelf.io.video import sample_frames

    video = _make_fake_video(tmp_path, n_frames=40, fps=20.0)
    # при интервале 500мс и 20fps шаг = 10 кадров → ~4 кадра из 40
    frames = list(sample_frames(video, interval_ms=500, adaptive=False))
    assert len(frames) == 4, f"Ожидали 4 кадра, получили {len(frames)}"


def test_sample_frames_timestamps_ascending(tmp_path):
    from shelf.io.video import sample_frames

    video = _make_fake_video(tmp_path, n_frames=60, fps=20.0)
    frames = list(sample_frames(video, interval_ms=200, adaptive=False))
    timestamps = [ts for ts, _ in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0)


def test_save_debug_frames(tmp_path):
    from shelf.io.video import save_debug_frames

    video = _make_fake_video(tmp_path, n_frames=50, fps=20.0)
    out_dir = tmp_path / "frames"
    saved = save_debug_frames(
        video, out_dir, interval_ms=500, max_frames=3, max_dim=160
    )
    assert len(saved) == 3
    assert all(p.exists() for p in saved)
    assert all(p.suffix == ".jpg" for p in saved)


def test_sample_frames_timestamps_are_milliseconds(tmp_path):
    from shelf.io.video import sample_frames

    video = _make_fake_video(tmp_path, n_frames=40, fps=20.0)
    frames = list(sample_frames(video, interval_ms=500, adaptive=False))
    timestamps = [ts for ts, _ in frames]
    assert timestamps[:4] == pytest.approx([0.0, 500.0, 1000.0, 1500.0])


def test_sample_frames_respects_max_timestamp_ms(tmp_path):
    from shelf.io.video import sample_frames

    video = _make_fake_video(tmp_path, n_frames=80, fps=20.0)
    frames = list(
        sample_frames(
            video,
            interval_ms=500,
            adaptive=False,
            max_timestamp_ms=750.0,
        )
    )
    assert [ts for ts, _ in frames] == pytest.approx([0.0, 500.0])


def test_mser_process_width_env_override(monkeypatch):
    from shelf.detect.detector import MSERDetector

    monkeypatch.setenv("SHELF_MSER_PROCESS_WIDTH", "480")
    assert MSERDetector().process_width == 480


def test_pipeline_max_tracks_env_keeps_best_scores(monkeypatch):
    from shelf.detect.detector import Detection
    from shelf.detect.tracker import TrackState
    from shelf.pipeline import _limit_tracks_for_ocr

    tracks = {
        1: TrackState(1, Detection(0, 0, 10, 10, 0.5), best_score=1.0),
        2: TrackState(2, Detection(0, 0, 10, 10, 0.5), best_score=5.0),
        3: TrackState(3, Detection(0, 0, 10, 10, 0.5), best_score=3.0),
    }
    monkeypatch.setenv("SHELF_MAX_TRACKS", "2")
    limited = _limit_tracks_for_ocr(tracks)
    assert list(limited) == [2, 3]
