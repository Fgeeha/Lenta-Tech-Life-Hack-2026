"""Evaluation scripts should fail gracefully when private videos are absent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_ceiling_reports_missing_data(tmp_path):
    json_out = tmp_path / "ceiling.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_ceiling.py",
            "--data-root",
            str(tmp_path / "missing_data"),
            "--ocr-engine",
            "none",
            "--json-out",
            str(json_out),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0
    assert "No ceiling videos" in result.stdout
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["rows"] == 0
    assert payload["missing"]


def test_eval_on_labeled_writes_missing_json(tmp_path):
    json_out = tmp_path / "labeled.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_on_labeled.py",
            "--data-root",
            str(tmp_path / "missing_data"),
            "--ocr-engine",
            "none",
            "--json-out",
            str(json_out),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0
    assert "No labeled videos" in result.stdout
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["rows"] == 0
    assert payload["overall"]["barcode_count"] == 0
    assert payload["missing"]
