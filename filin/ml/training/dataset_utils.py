from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def calculate_file_sha256(path: Path) -> str:
    """Возвращает SHA-256 файла для проверки различия наборов данных."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_run_ids(dataset: pd.DataFrame) -> set[str]:
    """Извлекает идентификаторы прогонов из metadata-колонки датасета."""
    if "run_id" not in dataset.columns:
        return set()
    return {str(value) for value in dataset["run_id"].dropna().unique()}
