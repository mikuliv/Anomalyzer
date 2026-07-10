from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from report_writer import write_evaluation_report
from train_baselines import calculate_metrics, load_dataset


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata модели не найдены: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_model(model_path: Path, dataset_path: Path, metadata_path: Path, report_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    metadata = load_metadata(metadata_path)
    df = load_dataset(dataset_path)
    target = metadata.get("target", "label")
    feature_columns = metadata.get("feature_columns") or []
    if target not in df.columns:
        raise ValueError(f"Целевая колонка не найдена в датасете: {target}")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют признаки модели: {', '.join(missing)}")

    X = df[feature_columns]
    y = df[target]
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    warning = None
    same_dataset = False
    metadata_dataset_path = metadata.get("dataset_path")
    try:
        same_dataset = Path(metadata.get("dataset_path", "")).resolve() == dataset_path.resolve()
        if same_dataset:
            warning = (
                "Оценка выполнена на том же датасете, который мог использоваться при обучении. "
                "Такой результат не является независимой проверкой качества."
            )
        else:
            warning = (
                "Оценка выполнена на отдельном датасете. Это более строгая проверка по сравнению с оценкой на том же наборе, "
                "но применимость результатов зависит от способа формирования датасета."
            )
    except OSError:
        warning = None

    write_evaluation_report(
        path=report_path,
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        model_name=str(metadata.get("model_name", "unknown")),
        metrics=metrics,
        warning=warning,
        metadata_dataset_path=str(metadata_dataset_path) if metadata_dataset_path else None,
        same_dataset=same_dataset,
        limitations=metadata.get("limitations", []),
    )
    return {"model_name": metadata.get("model_name"), "metrics": metrics, "report_path": str(report_path), "warning": warning}


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка сохранённой baseline-модели Филин на CSV-датасете.")
    parser.add_argument("--model", required=True, help="Путь к best_model.joblib.")
    parser.add_argument("--dataset", required=True, help="Путь к CSV-датасету.")
    parser.add_argument("--metadata", required=True, help="Путь к model_metadata.json.")
    parser.add_argument("--report", required=True, help="Путь к Markdown-отчёту.")
    args = parser.parse_args()
    result = evaluate_model(Path(args.model), Path(args.dataset), Path(args.metadata), Path(args.report))
    print(f"Модель оценена: {result['model_name']}")
    print(f"Отчёт сохранён: {result['report_path']}")
    if result["warning"]:
        print(result["warning"])


if __name__ == "__main__":
    main()
