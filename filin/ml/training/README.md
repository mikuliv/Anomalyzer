# Baseline training pipeline Филин

## Назначение

Модуль `filin/ml/training/` содержит первичный baseline pipeline для проверки обучения и оценки моделей на лабораторном датасете признаков.

Датасет v0.1 формируется из лабораторных/mock-событий. Он нужен для проверки ML pipeline, feature extraction, защиты от data leakage и формата будущей интеграции с backend. Он не является достаточным основанием для промышленной модели.

## Состав

- `split_dataset.py` - подготовка `X/y`, исключение metadata/leakage-полей и безопасный train/test split.
- `model_registry.py` - реестр baseline-моделей.
- `train_baselines.py` - обучение baseline-моделей и сохранение лучшей.
- `evaluate_model.py` - повторная оценка сохранённой модели.
- `report_writer.py` - формирование Markdown-отчётов.

## Модели

- `DummyClassifier` со стратегией `most_frequent`;
- `LogisticRegression` с `class_weight=balanced`;
- `RandomForestClassifier` с `class_weight=balanced`;
- `HistGradientBoostingClassifier`.

## Защита от data leakage

Из модельных признаков исключаются metadata и поля разметки:

- `run_id`;
- `run_sequence`;
- `scenario_id`;
- `window_start`;
- `window_end`;
- `planned_started_at`;
- `planned_finished_at`;
- `actual_started_at`;
- `actual_finished_at`;
- `label`;
- `label_type`;
- `mitre_technique_id`.

Также исключаются поля, содержащие `label`, `scenario`, `run_sequence`, `started_at` и `finished_at`.

Сначала выполняется train/test split. Все преобразования, включая imputing и scaling, обучаются только на train-части. SMOTE в baseline v0.1 не используется.

## Обучение

```powershell
python filin/ml/training/train_baselines.py --dataset filin/lab/output/datasets/windows_v0_1.csv --target label --output-dir filin/ml/artifacts/baseline_v0_1 --report filin/ml/reports/baseline_v0_1.md
```

Результат:

- `filin/ml/artifacts/baseline_v0_1/best_model.joblib`;
- `filin/ml/artifacts/baseline_v0_1/model_metadata.json`;
- `filin/ml/reports/baseline_v0_1.md`.

## Оценка

```powershell
python filin/ml/training/evaluate_model.py --model filin/ml/artifacts/baseline_v0_1/best_model.joblib --dataset filin/lab/output/datasets/windows_v0_1.csv --metadata filin/ml/artifacts/baseline_v0_1/model_metadata.json --report filin/ml/reports/evaluate_baseline_v0_1.md
```

Если оценка выполняется на том же датасете, который использовался при обучении, отчёт содержит предупреждение. Такой результат не является независимой проверкой качества.

## Оценка по разным laboratory runs

Для более честной проверки модель следует обучать на одном прогоне стенда, а оценивать на другом. Это снижает риск того, что модель выучит особенности одного конкретного `run_id` или расписания сценариев.

```powershell
python filin/ml/training/train_baselines.py --dataset filin/lab/output/datasets/windows_v0_1_run_001.csv --external-test-dataset filin/lab/output/datasets/windows_v0_1_run_002.csv --target label --output-dir filin/ml/artifacts/baseline_v0_1_external --report filin/ml/reports/baseline_v0_1_external.md
```

В этом режиме `--dataset` используется только для train, а `--external-test-dataset` только для test. Feature columns определяются по train dataset. Если в external test dataset не хватает признаков, обучение завершается понятной ошибкой. Лишние колонки test игнорируются.

Оценка на отдельном laboratory run является более строгой, чем случайный split внутри одного CSV, но всё ещё не подтверждает качество модели на реальном сетевом трафике, если оба набора сформированы в mock-режиме.

## Метрики

Accuracy не является основной метрикой для задач обнаружения инцидентов. Основное внимание уделяется macro/weighted F1, recall по attack-классам и confusion matrix.

## Ограничения v0.1

- Датасет построен на лабораторных/mock-событиях.
- Метрики не подтверждают качество модели на реальном сетевом трафике.
- Нейросетевые модели и финальное обучение планируются после расширения датасета и подключения реального сбора трафика.
