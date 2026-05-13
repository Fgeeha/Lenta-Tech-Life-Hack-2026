"""Деплой на Hugging Face Spaces (+ загрузка весов модели в HF Hub).

Использование:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    poetry run python scripts/deploy_hf.py
"""

import os
import subprocess
import sys
from pathlib import Path

SPACE_NAME = "shelf-control"
SPACE_SDK = "gradio"
MODEL_WEIGHTS = Path("models/pricetag_tiled_yolov8n.pt")
MODEL_REPO_SUFFIX = "shelf-pricetag-yolov8n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: установи токен: export HF_TOKEN=hf_xxxxx")
        print("Создать: https://huggingface.co/settings/tokens (Write access)")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo, upload_file, whoami

    api = HfApi(token=token)
    username = whoami(token=token)["name"]
    print(f"HF user: {username}")

    # --- 1. Загрузить модель в model repo ---
    model_repo_id = f"{username}/{MODEL_REPO_SUFFIX}"
    create_repo(model_repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    print(f"Model repo: https://huggingface.co/{model_repo_id}")

    if MODEL_WEIGHTS.exists():
        print(f"Загружаем {MODEL_WEIGHTS} в HF Hub...")
        upload_file(
            path_or_fileobj=str(MODEL_WEIGHTS),
            path_in_repo=MODEL_WEIGHTS.name,
            repo_id=model_repo_id,
            repo_type="model",
            token=token,
            commit_message="tiled YOLOv8n, mAP50=0.776",
        )
        print("Модель загружена!")
    else:
        print(f"WARN: {MODEL_WEIGHTS} не найден, пропускаем upload модели")

    # Обновляем HF_MODEL_REPO в yolo_sahi.py на реальный username
    sahi_path = Path("src/shelf/detect/yolo_sahi.py")
    content = sahi_path.read_text()
    old = 'HF_MODEL_REPO = "fgeeha/shelf-pricetag-yolov8n"'
    new = f'HF_MODEL_REPO = "{model_repo_id}"'
    if old in content and username != "fgeeha":
        sahi_path.write_text(content.replace(old, new))
        subprocess.run(["git", "add", str(sahi_path)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"chore: update HF model repo to {model_repo_id}"],
            check=True,
        )
        print(f"HF_MODEL_REPO обновлён: {model_repo_id}")

    # --- 2. Создать Space ---
    space_repo_id = f"{username}/{SPACE_NAME}"
    space_url = create_repo(
        space_repo_id, repo_type="space", space_sdk=SPACE_SDK,
        private=False, token=token, exist_ok=True,
    )
    print(f"Space: {space_url}")

    # --- 3. Push (без бинарных файлов — они уже исключены в .gitignore) ---
    hf_url = f"https://USER:{token}@huggingface.co/spaces/{space_repo_id}"
    result = subprocess.run(["git", "remote", "get-url", "hf"], capture_output=True)
    if result.returncode != 0:
        subprocess.run(["git", "remote", "add", "hf", hf_url], check=True)
    else:
        subprocess.run(["git", "remote", "set-url", "hf", hf_url], check=True)

    print("Пушим код на HF Spaces...")
    subprocess.run(["git", "push", "hf", "Master:main", "--force"], check=True)

    print(f"\nДеплой завершён!")
    print(f"Space: https://huggingface.co/spaces/{space_repo_id}")
    print(f"Model: https://huggingface.co/{model_repo_id}")
    print("\nОткрой Space в инкогнито и загрузи тестовое видео!")


if __name__ == "__main__":
    main()
