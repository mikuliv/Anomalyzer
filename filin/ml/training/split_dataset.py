from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from schema import get_forbidden_feature_columns, get_metadata_columns  # noqa: E402


LEAKAGE_MARKERS = ("label", "scenario", "run_sequence", "started_at", "finished_at")


def is_leakage_column(column: str, target_column: str) -> bool:
    if column == target_column:
        return True
    if column in set(get_metadata_columns()) | set(get_forbidden_feature_columns()):
        return True
    lowered = column.lower()
    return any(marker in lowered for marker in LEAKAGE_MARKERS)


def prepare_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка не найдена в датасете: {target_column}")
    if df.empty:
        raise ValueError("Датасет пустой.")

    excluded_columns: list[str] = []
    feature_columns: list[str] = []
    for column in df.columns:
        if is_leakage_column(column, target_column):
            excluded_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)
        else:
            excluded_columns.append(column)

    forbidden = sorted(set(feature_columns) & set(get_forbidden_feature_columns()))
    forbidden.extend(column for column in feature_columns if is_leakage_column(column, target_column))
    if forbidden:
        raise ValueError(f"В модельные признаки попали leakage-поля: {', '.join(sorted(set(forbidden)))}")
    if not feature_columns:
        raise ValueError("Не найдено числовых признаков для обучения.")

    return df[feature_columns], df[target_column], feature_columns, sorted(set(excluded_columns))


def validate_external_dataset_compatibility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> list[str]:
    warnings: list[str] = []
    if target_column not in train_df.columns:
        raise ValueError(f"В train dataset отсутствует целевая колонка: {target_column}")
    if target_column not in test_df.columns:
        raise ValueError(f"В external test dataset отсутствует целевая колонка: {target_column}")
    missing = [column for column in feature_columns if column not in test_df.columns]
    if missing:
        raise ValueError(f"В external test dataset отсутствуют признаки: {', '.join(missing)}")

    train_classes = set(train_df[target_column].dropna().astype(str))
    test_classes = set(test_df[target_column].dropna().astype(str))
    unknown_classes = sorted(test_classes - train_classes)
    if unknown_classes:
        warnings.append(
            "В external test dataset есть классы, которых не было в train dataset: "
            + ", ".join(unknown_classes)
            + ". Модель не сможет корректно предсказать такие классы, а метрики покажут ограничение текущего обучения."
        )
    return warnings


def safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, list[str]]:
    warnings: list[str] = []
    if len(X) < 4:
        raise ValueError("Слишком мало строк для train/test split. Нужно минимум 4 строки.")
    if y.nunique() < 2:
        raise ValueError("В датасете должен быть минимум один benign и один attack label.")

    class_counts = y.value_counts()
    use_stratify = bool((class_counts >= 2).all())
    if not use_stratify:
        warnings.append("В одном из классов меньше 2 объектов, stratify отключен.")

    try:
        split = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if use_stratify else None,
        )
        method = "stratified" if use_stratify else "plain"
        return (*split, method, warnings)
    except ValueError as error:
        warnings.append(f"Stratified split недоступен: {error}. Используется обычный train_test_split.")
        split = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
        return (*split, "plain", warnings)
