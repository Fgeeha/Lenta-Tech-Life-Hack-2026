"""Deploy to HF Space using upload_folder (atomic, no git, no LFS issues).

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    PYTHONPATH=src poetry run python scripts/deploy_hf.py

CI: called automatically by .github/workflows/hf-deploy.yml
    HF_TOKEN must be set as a GitHub Actions secret.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

SPACE_NAME = "shelf-control"
MODEL_EXCLUDE = {"RealESRGAN_x4plus.pth"}  # too large for Space (64 MB)
MODEL_MAX_BYTES = 20 * 1024 * 1024          # skip weights > 20 MB

INCLUDE = [
    "app.py",
    "requirements.txt",
    "src/",
    "data/catalog.csv",
]


def _strip_gradio_pin(req_path: Path) -> None:
    """Remove gradio line — HF Spaces injects its own pinned version."""
    lines = req_path.read_text().splitlines()
    filtered = [l for l in lines if not l.strip().startswith("gradio")]
    req_path.write_text("\n".join(filtered) + "\n")


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: set HF_TOKEN environment variable")
        print("Create at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo, restart_space, upload_folder

    api = HfApi(token=token)
    username = api.whoami()["name"]
    space_repo_id = f"{username}/{SPACE_NAME}"
    print(f"Deploying to: https://huggingface.co/spaces/{space_repo_id}")

    create_repo(
        space_repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        token=token,
        exist_ok=True,
    )

    root = Path(".")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for item in INCLUDE:
            src = root / item
            dst = tmp / item
            if src.is_dir():
                shutil.copytree(src, dst)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                print(f"  SKIP (not found): {item}")

        # YOLO weights: only small .pt files
        models_src = root / "models"
        if models_src.exists():
            models_dst = tmp / "models"
            models_dst.mkdir(exist_ok=True)
            for pt in sorted(models_src.glob("*.pt")):
                if pt.name in MODEL_EXCLUDE or pt.stat().st_size > MODEL_MAX_BYTES:
                    print(f"  SKIP model (excluded/too large): {pt.name}")
                    continue
                shutil.copy2(pt, models_dst / pt.name)
                print(f"  model: {pt.name} ({pt.stat().st_size // 1024} KB)")

        # Remove gradio pin — HF Spaces manages its own version
        req_dst = tmp / "requirements.txt"
        if req_dst.exists():
            _strip_gradio_pin(req_dst)

        print("Uploading to Space...")
        upload_folder(
            folder_path=str(tmp),
            repo_id=space_repo_id,
            repo_type="space",
            token=token,
            commit_message="deploy: update Space from CI",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        )

    print("Restarting Space...")
    try:
        restart_space(repo_id=space_repo_id, token=token)
        print("Space restarted.")
    except Exception as exc:
        print(f"restart_space: {exc} (Space will rebuild automatically)")

    print(f"\nDone: https://huggingface.co/spaces/{space_repo_id}")
    print("Build takes ~5-10 min (PaddleOCR + ultralytics are heavy).")


if __name__ == "__main__":
    main()
