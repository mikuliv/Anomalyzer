# Машинное обучение в проекте «Филин»

## Назначение

Раздел `filin/ml/` отвечает за подготовку признаков, обучение, оценку и будущий экспорт моделей обнаружения инцидентов информационной безопасности.

## Источники данных

Основной источник v0.1 - лабораторный pipeline:

```text
scenario_manifest.yaml -> execution_events.jsonl -> traffic_events.jsonl -> normalized_events.jsonl
```

В дальнейшем к нему должны добавиться события Zeek, Suricata и другие нормализованные источники.

## Pipeline подготовки данных

```text
raw events -> normalized events -> feature extraction -> windows.csv / flows.csv -> training
```

Сырые события не являются готовыми признаками для модели. Для обучения используются агрегированные window-level и flow-level датасеты.

## Feature extraction

Модуль `filin/ml/features/` содержит первый прототип построения признаков:

- `feature_catalog.yaml` - каталог признаков v0.1;
- `schema.py` - metadata и forbidden leakage columns;
- `validators.py` - базовая проверка CSV;
- `build_windows_dataset.py` - построение window-level датасета;
- `build_flows_dataset.py` - построение flow-level прототипа.

`validators.py` также проверяет учебные example CSV: наличие label, benign и attack label, числовые признаки, отсутствие бесконечных значений, leakage-поля и базовые арифметические соотношения.

## Feature catalog

Каталог признаков описывает назначение, источник, уровень агрегации и риск утечки разметки. Поля `scenario_id`, `run_sequence`, planned/actual time, `label`, `label_type` и `mitre_technique_id` не должны попадать в модельные признаки.

## Window-level dataset

Window-level датасет строится по временным окнам:

```powershell
python filin/ml/features/build_windows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/windows_v0_1.csv --window-seconds 60
```

Каждая строка содержит metadata и числовые признаки окна. Разметка берется из manifest.

## Flow-level dataset

Flow-level датасет v0.1 является прототипом:

```powershell
python filin/ml/features/build_flows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/flows_v0_1.csv
```

Он группирует normalized events по `source_role`, `target_role`, `scenario_id` и `event_type`. После подключения Zeek/Suricata этот слой должен быть расширен реальными сетевыми flow-полями.

Учебные examples CSV проверяются командами:

```powershell
python filin/ml/features/validators.py --csv filin/datasets/examples/windows_v0_1.example.csv --kind windows

python filin/ml/features/validators.py --csv filin/datasets/examples/flows_v0_1.example.csv --kind flows
```

## Обучение моделей

Планируемые семейства моделей:

- MLP;
- RandomForest;
- XGBoost/LightGBM;
- AutoEncoder;
- простые baseline-модели для сравнения.

Сначала выполняется train/test split, и только после этого допускается балансировка классов и обучение scaler/encoder только на train-части.

## Baseline training/evaluation pipeline

На текущем этапе baseline-модели обучаются на лабораторном датасете v0.1 только для проверки корректности ML-контура. Нейросетевые модели и финальное обучение планируются после расширения датасета и подключения реального сбора трафика.

Обучение baseline-моделей:

```powershell
python filin/ml/training/train_baselines.py --dataset filin/lab/output/datasets/windows_v0_1.csv --target label --output-dir filin/ml/artifacts/baseline_v0_1 --report filin/ml/reports/baseline_v0_1.md
```

Оценка сохранённой модели:

```powershell
python filin/ml/training/evaluate_model.py --model filin/ml/artifacts/baseline_v0_1/best_model.joblib --dataset filin/lab/output/datasets/windows_v0_1.csv --metadata filin/ml/artifacts/baseline_v0_1/model_metadata.json --report filin/ml/reports/evaluate_baseline_v0_1.md
```

Артефакты `filin/ml/artifacts/` и отчёты `filin/ml/reports/` являются локальными результатами прогонов и не хранятся в Git.

## Оценка по разным laboratory runs

Для более честной проверки модель следует обучать на одном прогоне стенда, а оценивать на другом. Это снижает риск того, что модель выучит особенности одного конкретного `run_id` или расписания сценариев.

```powershell
# Run 001
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_001 --base-time 2026-07-09T13:00:00Z --gap-seconds 30 --repeat 1 --mock --window-seconds 60

# Run 002
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_002 --base-time 2026-07-10T13:00:00Z --gap-seconds 45 --repeat 1 --mock --window-seconds 60

# Обучение на run_001 и external-test на run_002
python filin/ml/training/run_external_experiment.py --train-run run_001 --test-run run_002 --target label
```

Прямой вызов training CLI:

```powershell
python filin/ml/training/train_baselines.py --dataset filin/lab/output/datasets/windows_v0_1_run_001.csv --external-test-dataset filin/lab/output/datasets/windows_v0_1_run_002.csv --target label --output-dir filin/ml/artifacts/baseline_v0_1_external --report filin/ml/reports/baseline_v0_1_external.md
```

Оценка на отдельном laboratory run является более строгой, чем случайный split внутри одного CSV. Если оба набора сформированы в mock-режиме, такая проверка всё ещё не подтверждает качество модели на реальном сетевом трафике.

## Оценка качества

Accuracy не является основной метрикой для задач обнаружения инцидентов. Важны precision, recall, F1, confusion matrix, ROC-AUC/PR-AUC и анализ ошибок по классам.

Отдельно нужно оценивать:

- ложные срабатывания на benign-активности;
- пропуски attack-сценариев;
- устойчивость модели к изменению расписания;
- качество по каждому классу инцидента.

## Предотвращение data leakage

Metadata используется для разметки и анализа, но не для обучения модели. Запрещенные для признаков поля:

- `scenario_id`;
- `run_sequence`;
- `planned_started_at`;
- `planned_finished_at`;
- `actual_started_at`;
- `actual_finished_at`;
- `label`;
- `label_type`;
- `mitre_technique_id`.

## Ограничения v0.1

- Mock-события являются синтетическими и подходят для проверки pipeline.
- Для итогового обучения нужен реальный сбор трафика в Docker/VMware-стенде.
- Flow-level датасет пока не является полноценной заменой Zeek conn/http/dns логов.
- Feature catalog v0.1 будет уточняться после первых экспериментов.

## План развития

- Подключить Zeek/Suricata как источники flow и application logs.
- Расширить feature catalog и схемы датасетов.
- Расширить train/test pipeline и версионирование экспериментов.
- Сравнить baseline с более сложными моделями.
- Подготовить экспорт выбранной модели в ONNX.
