# ML-подсистема

## Назначение

Построение проверяемых feature datasets, внешняя оценка и диагностические analyses лабораторного сетевого сенсора.

## Что реализовано

`features/` содержит profiles, builders и validators; `analysis/` — audits; `experiments/` — v0.3.1 baseline и v0.3.2 frozen robustness evaluation. `training/` содержит загрузчики и контроль разделения данных.

## Входные данные и выходные данные

Входом sensor pipeline являются correlated Zeek observations. Datasets, reports и model artifacts являются runtime artifacts и не коммитятся.

## Запуск

Для быстрой проверки: `python -m unittest discover -s filin/ml/tests -p "test_*.py"`.

## Проверки

Model selection, external evaluation и robustness evaluation разделены. В v0.3.2 запрещено `.fit()` на robustness data.

## Ограничения

ML-результаты относятся к controlled laboratory data; backend model integration не начата.

## Связанные документы

[Эксперименты](../docs/experiments.md), [datasets](../datasets/README.md), [ограничения](../docs/limitations.md).
