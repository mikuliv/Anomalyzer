# Robustness laboratory

## Назначение

Подготовка независимых external runs для проверки устойчивости frozen sensor baseline.

## Что реализовано

v0.3.2 использует 12 completed runs: по три topology, background, temporal и combined. Эти данные применяются только к predict/evaluation.

## Основные файлы

`run_v0_3_2_stage.py` координирует integrity audits и evaluation pipeline.

## Входные данные и выходные данные

Вход — существующая campaign и frozen model manifest. Выход — runtime reports v0.3.2.

## Запуск

`python filin/lab/robustness/run_v0_3_2_stage.py --help`

## Проверки

Runner с `--resume` должен распознавать успешные runs и не выполнять Docker campaign повторно.

## Ограничения

Положительный результат policy не означает production readiness.

## Связанные документы

[Эксперименты](../../docs/experiments.md), [ограничения](../../docs/limitations.md).
