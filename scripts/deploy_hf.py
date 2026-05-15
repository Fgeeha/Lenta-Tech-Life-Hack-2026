"""Деплой на Hugging Face Spaces через huggingface_hub API (без git push).

Использование:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    poetry run python scripts/deploy_hf.py
"""

import os
import sys
from pathlib import Path

SPACE_NAME = "shelf-control"
SPACE_SDK = "gradio"
MODEL_WEIGHTS = Path("models/pricetag_tiled_yolov8n.pt")
MODEL_REPO_SUFFIX = "shelf-pricetag-yolov8n"

# Файлы, которые нужно залить в Space
SPACE_FILES = [
    "app.py",
    "requirements.txt",
    "src/shelf/__init__.py",
    "src/shelf/schema.py",
    "src/shelf/pipeline.py",
    "src/shelf/io/__init__.py",
    "src/shelf/io/video.py",
    "src/shelf/io/writer.py",
    "src/shelf/detect/__init__.py",
    "src/shelf/detect/detector.py",
    "src/shelf/detect/tracker.py",
    "src/shelf/detect/yolo_sahi.py",
    "src/shelf/ocr/__init__.py",
    "src/shelf/ocr/engine.py",
    "src/shelf/ocr/parser.py",
    "src/shelf/ocr/preprocess.py",
    "src/shelf/ocr/template.py",
    "src/shelf/postproc/__init__.py",
    "src/shelf/postproc/merge.py",
    "src/shelf/qr/__init__.py",
    "src/shelf/qr/decoder.py",
    "src/shelf/ui/__init__.py",
    "src/shelf/ui/gradio_app.py",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: export HF_TOKEN=hf_xxxxx")
        print("Создать: https://huggingface.co/settings/tokens")
        sys.exit(1)

    from huggingface_hub import create_repo, upload_file, whoami

    # api = HfApi(token=token)
    username = whoami(token=token)["name"]
    print(f"HF user: {username}")

    # --- 1. Модель → отдельный model repo ---
    model_repo_id = f"{username}/{MODEL_REPO_SUFFIX}"
    create_repo(
        model_repo_id,
        repo_type="model",
        private=False,
        token=token,
        exist_ok=True,
    )
    if MODEL_WEIGHTS.exists():
        print(f"Загружаем {MODEL_WEIGHTS}...")
        upload_file(
            path_or_fileobj=str(MODEL_WEIGHTS),
            path_in_repo=MODEL_WEIGHTS.name,
            repo_id=model_repo_id,
            repo_type="model",
            token=token,
            commit_message="tiled YOLOv8n mAP50=0.776",
        )
        print(f"Модель: https://huggingface.co/{model_repo_id}")

    # --- 2. Обновить HF_MODEL_REPO в yolo_sahi.py ---
    sahi_path = Path("src/shelf/detect/yolo_sahi.py")
    content = sahi_path.read_text()
    if (
        f'HF_MODEL_REPO = "fgeeha/{MODEL_REPO_SUFFIX}"' in content
        and username != "fgeeha"
    ):
        sahi_path.write_text(
            content.replace(
                f'HF_MODEL_REPO = "fgeeha/{MODEL_REPO_SUFFIX}"',
                f'HF_MODEL_REPO = "{model_repo_id}"',
            )
        )
        print(f"HF_MODEL_REPO обновлён: {model_repo_id}")

    # --- 3. Space → upload файлов через API (без git, без истории) ---
    space_repo_id = f"{username}/{SPACE_NAME}"
    create_repo(
        space_repo_id,
        repo_type="space",
        space_sdk=SPACE_SDK,
        private=False,
        token=token,
        exist_ok=True,
    )
    print(f"Space: https://huggingface.co/spaces/{space_repo_id}")

    missing = [f for f in SPACE_FILES if not Path(f).exists()]
    if missing:
        print(f"WARN: файлы не найдены: {missing}")

    print("Загружаем файлы в Space...")
    for rel_path in SPACE_FILES:
        p = Path(rel_path)
        if not p.exists():
            continue
        upload_file(
            path_or_fileobj=str(p),
            path_in_repo=rel_path,
            repo_id=space_repo_id,
            repo_type="space",
            token=token,
        )
        print(f"  ✓ {rel_path}")

    print("\nДеплой завершён!")
    print(f"Space:  https://huggingface.co/spaces/{space_repo_id}")
    print(f"Model:  https://huggingface.co/{model_repo_id}")
    print(
        "\nОткрой Space в инкогнито, дождись сборки (~3 мин) и загрузи видео!"
    )


if __name__ == "__main__":
    main()
