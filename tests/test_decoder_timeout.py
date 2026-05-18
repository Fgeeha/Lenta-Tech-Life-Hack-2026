"""Tests for SIGALRM-based timeout on WeChatQR and QReader."""

import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import shelf.qr.decoder as decoder


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="SIGALRM not available")
def test_wechat_qr_timeout_returns_empty(monkeypatch):
    """When WeChatQR hangs longer than timeout, _try_wechat_qr returns [] without crash."""
    mock_reader = MagicMock()

    def slow_decode(image):
        time.sleep(5)  # longer than timeout=1
        return (["payload"], None)

    mock_reader.detectAndDecode = slow_decode
    monkeypatch.setattr(decoder, "_WECHAT_QR", mock_reader)
    monkeypatch.setattr(decoder, "_WECHAT_TIMEOUT", 1.0)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    t0 = time.time()
    result = decoder._try_wechat_qr(img)
    elapsed = time.time() - t0

    assert result == []
    assert elapsed < 2.5, f"Timeout did not fire: took {elapsed:.1f}s"


def test_wechat_qr_normal_decode_works(monkeypatch):
    """Fast successful decode returns result without triggering timeout."""
    mock_reader = MagicMock()
    mock_reader.detectAndDecode = MagicMock(return_value=(["test_payload"], None))
    monkeypatch.setattr(decoder, "_WECHAT_QR", mock_reader)
    monkeypatch.setattr(decoder, "_WECHAT_TIMEOUT", 2.0)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = decoder._try_wechat_qr(img)

    assert result == ["test_payload"]


def test_qreader_cached_across_calls(monkeypatch):
    """QReader() must not be recreated on each call — it is cached."""
    monkeypatch.setattr(decoder, "_QREADER", None)
    create_count = [0]

    class FakeQReader:
        def __init__(self):
            create_count[0] += 1

        def detect_and_decode(self, image):
            return ["x"]

    fake_module = MagicMock()
    fake_module.QReader = FakeQReader
    monkeypatch.setitem(sys.modules, "qreader", fake_module)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    decoder._try_qreader(img)
    decoder._try_qreader(img)
    decoder._try_qreader(img)

    assert create_count[0] == 1, f"QReader created {create_count[0]} times instead of once"
