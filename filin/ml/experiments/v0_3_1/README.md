# Baseline evaluation v0.3.1

## Назначение

Независимая внешняя оценка профилей `client_core_v0_2` и `network_sensor_v0_3` без feature fusion.

## Что реализовано

Model selection выполнялся только по шести независимым train-runs с Leave-One-Train-Run-Out. Три test-runs применялись один раз для external evaluation. По результатам рекомендован `network_sensor_v0_3`.

## Основные файлы

Runtime evaluation report хранится в `filin/ml/reports/v0_3_1/` и не коммитится.

## Входные данные и выходные данные

Используются проверенные client и sensor datasets с независимыми campaign roles. Output — model artifacts и evaluation reports вне Git.

## Запуск

Доступные параметры scripts проверяются через `--help`; повторная оценка не требуется для проверки документации.

## Проверки

Pooled test: `client_core_v0_2` macro F1 `0.024`; `network_sensor_v0_3` macro F1 `0.918`, balanced accuracy `0.972`, attack macro recall `1.000`.

## Ограничения

Результаты относятся к controlled laboratory environment. Backend model integration отсутствует.

## Связанные документы

[Эксперименты](../../../docs/experiments.md), [v0.3.2 robustness evaluation](../v0_3_2/README.md).
