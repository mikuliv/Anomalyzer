# Robustness evaluation v0.3.2

## Назначение

Внешняя проверка устойчивости зафиксированного baseline `network_sensor_v0_3`.

## Что реализовано

Frozen `LogisticRegression` с `SimpleImputer(strategy="median")` и `StandardScaler` оценивается без `.fit()` на 12 независимых robustness-runs: topology, background, temporal и combined.

## Основные файлы

- `frozen_model_manifest.yaml` — manifest зафиксированной модели;
- `robustness_policy.yaml` — неизменная laboratory policy;
- `run_robustness_evaluation.py` — evaluation без retraining.

## Входные и выходные данные

Вход — runtime datasets `network_sensor_v0_3` и artifact модели v0.3.1. Выход — runtime reports в `filin/ml/reports/v0_3_2/`; они не коммитятся.

## Запуск

Проверка доступных параметров: `python filin/ml/experiments/v0_3_2/run_robustness_evaluation.py --help`.

## Проверки и ограничения

Robustness-данные не участвуют в выборе модели, preprocessing или tuning. Результаты подтверждают только controlled laboratory policy; backend integration не начата.

## Связанные документы

[Эксперименты](../../../docs/experiments.md), [ограничения](../../../docs/limitations.md), [воспроизводимость](../../../docs/reproducibility.md).
