"""Создать HF Space и запушить репозиторий.

Использование:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    poetry run python scripts/deploy_hf.py
"""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

SPACE_NAME = "shelf-control"
SPACE_SDK = "gradio"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("Установи токен: export HF_TOKEN=hf_xxxxx")
        print("Токен создать на: https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=token)
    user = whoami(token=token)
    username = user["name"]
    repo_id = f"{username}/{SPACE_NAME}"
    print(f"Пользователь HF: {username}")

    # Создать Space (если не существует)
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=SPACE_SDK,
            private=False,
            token=token,
            exist_ok=True,
        )
        print(f"Space: {url}")
    except Exception as e:
        print(f"create_repo: {e}")

    # Добавить remote
    hf_url = f"https://USER:{token}@huggingface.co/spaces/{repo_id}"
    result = subprocess.run(["git", "remote", "get-url", "hf"], capture_output=True)
    if result.returncode != 0:
        subprocess.run(["git", "remote", "add", "hf", hf_url], check=True)
        print("Remote 'hf' добавлен")
    else:
        subprocess.run(["git", "remote", "set-url", "hf", hf_url], check=True)
        print("Remote 'hf' обновлён")

    # Push (может занять время из-за весов модели)
    print("Пушим на HF Spaces (модель ~6 МБ, может занять минуту)...")
    subprocess.run(["git", "push", "hf", "Master:main", "--force"], check=True)
    print(f"\nДеплой завершён: https://huggingface.co/spaces/{repo_id}")
    print("Открой в инкогнито-режиме и проверь загрузку видео!")


if __name__ == "__main__":
    main()
