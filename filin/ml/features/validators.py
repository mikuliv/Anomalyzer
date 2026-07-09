from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from schema import get_forbidden_feature_columns, get_model_feature_columns


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"CSV-файл не найден: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_dataset(path: Path) -> None:
    columns, rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV-файл пустой: {path}")
    if "label" not in columns:
        raise ValueError("В CSV отсутствует обязательная колонка label.")

    labels = {row.get("label") for row in rows}
    has_benign = "benign" in labels
    has_attack = any(label not in {None, "", "benign", "unknown"} for label in labels)
    if not has_benign or not has_attack:
        raise ValueError("В датасете должен быть хотя бы один benign и один attack label.")

    model_features = get_model_feature_columns(columns)
    forbidden = sorted(set(model_features) & set(get_forbidden_feature_columns()))
    if forbidden:
        raise ValueError(f"Запрещенные leakage-поля попали в признаки модели: {', '.join(forbidden)}")

    for row_number, row in enumerate(rows, start=2):
        for feature in model_features:
            value = row.get(feature, "")
            if value == "":
                raise ValueError(f"Пустое значение признака {feature} в строке {row_number}.")
            try:
                numeric = float(value)
            except ValueError as error:
                raise ValueError(f"Признак {feature} не приводится к числу в строке {row_number}.") from error
            if not math.isfinite(numeric):
                raise ValueError(f"Признак {feature} имеет бесконечное значение в строке {row_number}.")


def print_validation_result(path: Path) -> None:
    validate_dataset(path)
    print(f"Проверка датасета пройдена: {path}")
