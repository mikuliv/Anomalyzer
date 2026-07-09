# Построение признаков Филин

Модуль `filin/ml/features` преобразует нормализованные лабораторные события в агрегированные датасеты признаков для будущего обучения моделей.

Сырые события не являются готовыми признаками для модели. Для обучения используются агрегированные window-level и flow-level датасеты:

```text
raw events -> normalized events -> feature extraction -> windows.csv / flows.csv -> training
```

## Каталог признаков

`feature_catalog.yaml` описывает признаки v0.1, их уровень агрегации, источник, назначение и риск утечки разметки.

Поля `scenario_id`, `run_sequence`, `planned_started_at`, `planned_finished_at`, `actual_started_at`, `actual_finished_at`, `label`, `label_type` и `mitre_technique_id` используются только как metadata, для разметки и анализа. Они не являются входными признаками модели.

## Window-level датасет

Команда строит временные окна фиксированной длины и считает признаки по событиям внутри каждого окна:

```powershell
python filin/ml/features/build_windows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/windows_v0_1.csv --window-seconds 60
```

Разметка берется из manifest. Если окно пересекается с attack-сценарием, используется label этого сценария. Если окно пересекается только с benign-сценарием, используется `benign`. Если окно не попадает ни в один сценарий, используется `unknown`.

## Flow-level датасет

В v0.1 flow-level датасет является прототипом. Он группирует нормализованные лабораторные события по `source_role`, `target_role`, `scenario_id` и `event_type`, а затем считает базовые агрегаты:

```powershell
python filin/ml/features/build_flows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/flows_v0_1.csv
```

После подключения Zeek/Suricata этот датасет должен быть расширен реальными сетевыми flow-полями.

## Проверка

Оба сборщика вызывают базовый валидатор CSV:

- CSV не должен быть пустым;
- должна быть колонка `label`;
- должны присутствовать benign и attack label;
- leakage-поля не должны попадать в список модельных признаков;
- все модельные признаки должны приводиться к числам;
- бесконечные значения запрещены.
