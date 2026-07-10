from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model_registry import MODEL_PRIORITY, build_baseline_models
from report_writer import write_training_report
from split_dataset import prepare_xy, safe_train_test_split


LIMITATIONS = [
    "Модель обучена на лабораторном mock-датасете v0.1 и не предназначена для промышленного применения.",
    "Полученные метрики проверяют корректность ML pipeline, а не качество на реальном сетевом трафике.",
    "SMOTE в baseline v0.1 не используется.",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Датасет не найден. Сначала сформируйте windows_v0_1.csv через build_windows_dataset.py.")
    df = pd.read_csv(path, encoding="utf-8")
    if df.empty:
        raise ValueError("Датасет пустой.")
    return df


def calculate_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "classification_report_text": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def select_best_model(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    if not metrics_by_model:
        raise ValueError("Ни одна baseline-модель не была успешно обучена.")
    best_score = max(metrics["macro_f1"] for metrics in metrics_by_model.values())
    candidates = {name for name, metrics in metrics_by_model.items() if metrics["macro_f1"] == best_score}
    for model_name in MODEL_PRIORITY:
        if model_name in candidates:
            return model_name
    return sorted(candidates)[0]


def train_baselines(
    dataset_path: Path,
    target: str,
    output_dir: Path,
    report_path: Path,
    test_size: float,
    random_state: int,
    min_class_count: int,
) -> dict[str, Any]:
    df = load_dataset(dataset_path)
    X, y, feature_columns, excluded_columns = prepare_xy(df, target)
    class_distribution = {str(label): int(count) for label, count in y.value_counts().sort_index().items()}
    low_count = {label: count for label, count in class_distribution.items() if count < min_class_count}
    split_warnings: list[str] = []
    if low_count:
        split_warnings.append(f"Есть классы с числом объектов меньше min_class_count={min_class_count}: {low_count}")

    X_train, X_test, y_train, y_test, split_method, warnings = safe_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    split_warnings.extend(warnings)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    model_errors: dict[str, str] = {}
    for model_name, model in build_baseline_models(random_state).items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics_by_model[model_name] = calculate_metrics(y_test, y_pred)
            trained_models[model_name] = model
        except Exception as error:
            model_errors[model_name] = str(error)

    best_model_name = select_best_model(metrics_by_model)
    best_model = trained_models[best_model_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metadata_path = output_dir / "model_metadata.json"
    joblib.dump(best_model, model_path)

    metadata = {
        "dataset_path": str(dataset_path),
        "target": target,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "model_name": best_model_name,
        "metrics": metrics_by_model[best_model_name],
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "limitations": LIMITATIONS,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    split_info = {
        "test_size": test_size,
        "random_state": random_state,
        "method": split_method,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "warnings": split_warnings,
    }
    write_training_report(
        path=report_path,
        dataset_path=str(dataset_path),
        target=target,
        feature_columns=feature_columns,
        excluded_columns=excluded_columns,
        class_distribution=class_distribution,
        split_info=split_info,
        metrics_by_model=metrics_by_model,
        model_errors=model_errors,
        best_model=best_model_name,
        limitations=LIMITATIONS + split_warnings,
    )
    return {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_path),
        "metrics": metrics_by_model[best_model_name],
        "model_errors": model_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение baseline-моделей Филин на лабораторном датасете признаков.")
    parser.add_argument("--dataset", required=True, help="Путь к CSV-датасету признаков.")
    parser.add_argument("--target", default="label", help="Целевая колонка.")
    parser.add_argument("--output-dir", required=True, help="Папка для сохранения модели и metadata.")
    parser.add_argument("--report", required=True, help="Путь к Markdown-отчёту.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Доля test-части.")
    parser.add_argument("--random-state", type=int, default=42, help="Фиксированное состояние генератора.")
    parser.add_argument("--min-class-count", type=int, default=2, help="Минимальное число объектов класса для предупреждения.")
    args = parser.parse_args()

    result = train_baselines(
        dataset_path=Path(args.dataset),
        target=args.target,
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        test_size=args.test_size,
        random_state=args.random_state,
        min_class_count=args.min_class_count,
    )
    print(f"Лучшая модель: {result['best_model']}")
    print(f"Модель сохранена: {result['model_path']}")
    print(f"Metadata сохранены: {result['metadata_path']}")
    print(f"Отчёт сохранён: {result['report_path']}")


if __name__ == "__main__":
    main()
