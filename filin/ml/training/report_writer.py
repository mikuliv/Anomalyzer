from __future__ import annotations

from pathlib import Path
from typing import Any


METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def matrix_to_markdown(matrix: list[list[int]], labels: list[str]) -> str:
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    separator = "| --- | " + " | ".join("---" for _ in labels) + " |"
    rows = [header, separator]
    for label, values in zip(labels, matrix):
        rows.append("| " + label + " | " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def metrics_table(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    rows = ["| Модель | " + " | ".join(METRIC_ORDER) + " |"]
    rows.append("| --- | " + " | ".join("---" for _ in METRIC_ORDER) + " |")
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            "| "
            + model_name
            + " | "
            + " | ".join(format_metric(metrics.get(metric, "")) for metric in METRIC_ORDER)
            + " |"
        )
    return "\n".join(rows)


def write_training_report(
    path: Path,
    dataset_path: str,
    training_datasets: list[str],
    training_dataset_sha256: dict[str, str],
    external_test_dataset_sha256: str | None,
    external_test_dataset_path: str | None,
    target: str,
    feature_columns: list[str],
    excluded_columns: list[str],
    class_distribution: dict[str, int],
    test_class_distribution: dict[str, int] | None,
    split_info: dict[str, Any],
    compatibility_info: dict[str, Any],
    metrics_by_model: dict[str, dict[str, Any]],
    model_errors: dict[str, str],
    best_model: str,
    limitations: list[str],
) -> None:
    lines: list[str] = [
        "# Отчёт по baseline-моделям Филин v0.1",
        "",
        "## Назначение",
        "",
        "Отчёт фиксирует первичную проверку ML pipeline на лабораторном датасете признаков.",
        "",
        "## Режим оценки",
        "",
        f"- Режим оценки: {split_info.get('evaluation_mode')}",
        "",
        "## Использованный датасет",
        "",
        f"- Путь: `{dataset_path}`",
        "",
        "## Train dataset",
        "",
        f"- Путь: `{dataset_path}`",
        f"- Строк: {split_info.get('train_source_rows')}",
        *[f"- Дополнительный train dataset: `{path}`" for path in training_datasets[1:]],
        *[f"- SHA-256 `{path}`: `{digest}`" for path, digest in training_dataset_sha256.items()],
        "",
        "## External test dataset",
        "",
        f"- Путь: `{external_test_dataset_path or 'не используется'}`",
        f"- Строк: {split_info.get('external_test_rows') or 'не используется'}",
        f"- SHA-256: `{external_test_dataset_sha256 or 'не используется'}`",
        "",
        "## Ограничения датасета",
        "",
        "Датасет v0.1 сформирован на основе лабораторных/mock-событий. Полученные метрики не являются подтверждением качества модели на реальном сетевом трафике. На данном этапе обучение используется для проверки корректности ML pipeline.",
        "",
        "## Целевая переменная",
        "",
        f"- `{target}`",
        "",
        "## Исключённые metadata/leakage поля",
        "",
        "Metadata columns excluded:",
        "",
        *[f"- `{column}`" for column in excluded_columns],
        "",
        "Forbidden leakage columns excluded:",
        "",
        "- `run_id`",
        "- `run_sequence`",
        "- `scenario_id`",
        "- `window_start`",
        "- `window_end`",
        "- `planned_started_at`",
        "- `planned_finished_at`",
        "- `actual_started_at`",
        "- `actual_finished_at`",
        "- `label`",
        "- `label_type`",
        "- `mitre_technique_id`",
        "",
        "## Использованные признаки",
        "",
        "Model feature columns:",
        "",
        *[f"- `{column}`" for column in feature_columns],
        "",
        "## Распределение классов",
        "",
        "Train dataset:",
        "",
        *[f"- `{label}`: {count}" for label, count in class_distribution.items()],
        "",
        "## Сравнение распределения классов",
        "",
        "Test dataset:",
        "",
        *[f"- `{label}`: {count}" for label, count in (test_class_distribution or {}).items()],
        "",
        "## Совместимость признаков",
        "",
        f"- Признаков в train: {len(feature_columns)}",
        f"- Все признаки найдены в test: {compatibility_info.get('all_features_present', True)}",
        f"- Лишние колонки test игнорируются: {compatibility_info.get('extra_test_columns', 0)}",
        "",
        "## Методика разделения train/test",
        "",
        f"- test_size: {split_info.get('test_size')}",
        f"- random_state: {split_info.get('random_state')}",
        f"- split_method: {split_info.get('method')}",
        f"- train_rows: {split_info.get('train_rows')}",
        f"- test_rows: {split_info.get('test_rows')}",
        "",
        "Все преобразования, включая imputing и scaling, обучаются только на train-части.",
        "",
        "## Модели",
        "",
        "- DummyClassifier",
        "- LogisticRegression",
        "- RandomForestClassifier",
        "- HistGradientBoostingClassifier",
        "",
        "## Метрики",
        "",
        "Accuracy не является основной метрикой для задач обнаружения инцидентов. Основное внимание уделяется macro/weighted F1, recall по attack-классам и confusion matrix.",
        "Высокое значение accuracy при низком macro F1 может быть связано с преобладанием benign-класса и не означает устойчивого обнаружения всех типов лабораторных инцидентов.",
        "",
        "## Метрики на external test",
        "",
        "Если режим оценки использует external test dataset, метрики ниже рассчитаны на отдельном laboratory run.",
        "",
        metrics_table(metrics_by_model),
        "",
        "## Confusion matrix",
        "",
    ]

    for model_name, metrics in metrics_by_model.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                matrix_to_markdown(metrics["confusion_matrix"], metrics["labels"]),
                "",
                "Classification report:",
                "",
                "```text",
                metrics["classification_report_text"],
                "```",
                "",
            ]
        )

    if model_errors:
        lines.extend(["## Ошибки моделей", ""])
        lines.extend(f"- `{model}`: {error}" for model, error in model_errors.items())
        lines.append("")

    lines.extend(
        [
            "## Лучшая модель",
            "",
            f"- `{best_model}`",
            "",
            "## Выводы",
            "",
            "Baseline pipeline выполнил train/test split, обучил доступные модели и сохранил лучшую модель по macro_f1.",
            "",
            "## Ограничения и дальнейшие шаги",
            "",
            *[f"- {item}" for item in limitations],
            "- Оценка на отдельном laboratory run является более строгой, чем случайный split внутри одного CSV, но всё ещё не подтверждает качество модели на реальном сетевом трафике, если оба набора сформированы в mock-режиме.",
            "- Подключить реальные Zeek/Suricata события.",
            "- Повторить оценку на независимом тестовом наборе.",
            "- Проверить recall по attack-классам и причины ошибок.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_evaluation_report(
    path: Path,
    dataset_path: str,
    model_path: str,
    model_name: str,
    metrics: dict[str, Any],
    warning: str | None,
    metadata_dataset_path: str | None,
    same_dataset: bool,
    limitations: list[str],
) -> None:
    lines = [
        "# Отчёт по оценке baseline-модели Филин v0.1",
        "",
        "## Назначение",
        "",
        "Отчёт фиксирует повторную оценку сохранённой baseline-модели на указанном CSV.",
        "",
        "## Модель",
        "",
        f"- Путь: `{model_path}`",
        f"- Имя: `{model_name}`",
        "",
        "## Датасет",
        "",
        f"- Путь: `{dataset_path}`",
        f"- Dataset path из metadata модели: `{metadata_dataset_path or 'не указан'}`",
        f"- Evaluation dataset совпадает с training dataset: `{same_dataset}`",
        "",
    ]
    if warning:
        lines.extend(["## Предупреждение", "", warning, ""])
    lines.extend(
        [
            "## Метрики",
            "",
            metrics_table({model_name: metrics}),
            "",
            "## Confusion matrix",
            "",
            matrix_to_markdown(metrics["confusion_matrix"], metrics["labels"]),
            "",
            "## Classification report",
            "",
            "```text",
            metrics["classification_report_text"],
            "```",
            "",
            "## Ограничения",
            "",
            *[f"- {item}" for item in limitations],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
