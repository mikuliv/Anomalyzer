# Анализы и audits

## Назначение

Проверка capture, correlation, aggregation, feature availability, provenance, split и диагностик экспериментов.

## Что реализовано

Audits проверяют PCAP/Zeek integrity, marker-aware assignments, profile/validator rules, feature consistency и разделение campaign roles.

## Основные файлы

Здесь находятся v0.2 window audits, v0.3 sensor audits и v0.3.2 analyses robustness evaluation.

## Входные данные и выходные данные

Вход — runtime campaign artifacts; output — JSON/Markdown reports в `filin/ml/reports/`, не коммитятся.

## Запуск

Параметры конкретного audit проверяются через `python <script> --help`.

## Проверки

Агрегация пересчитывается из тех же functions, что используют builders; `NaN` не подменяется нулём.

## Ограничения

Audit success подтверждает техническую целостность лабораторного pipeline, а не промышленную готовность.

## Связанные документы

[Архитектура](../../docs/architecture.md), [ограничения](../../docs/limitations.md).
