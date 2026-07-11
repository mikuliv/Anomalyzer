# Загрузка и оценка ML-данных

## Назначение

Загрузка datasets, контроль train/test separation и подготовка данных для экспериментов.

## Что реализовано

Loaders проверяют profile consistency, ordered feature list, hashes и исключают metadata из `X`.

## Входные данные и выходные данные

Вход — проверенные runtime datasets. Выход — in-memory matrices и experiment reports.

## Запуск

CLI и параметры конкретных scripts проверяйте через `--help`.

## Проверки

Для v0.3.1 model selection разрешён только на train-runs. Для v0.3.2 frozen model применяется к robustness data без retraining.

## Ограничения

Этот каталог не выполняет backend integration и не является online inference service.

## Связанные документы

[Эксперименты](../../docs/experiments.md), [воспроизводимость](../../docs/reproducibility.md).
