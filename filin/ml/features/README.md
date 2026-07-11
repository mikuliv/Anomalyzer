# Признаки и datasets

## Назначение

Определение feature profiles, aggregation по scenario execution windows и profile-aware validation.

## Что реализовано

`client_core_v0_2` и `client_extended_v0_2` используют client observations. `network_sensor_v0_3` использует только correlated Zeek observations. Окна формируются по actual execution intervals; metadata и identifiers не являются model features.

## Основные файлы

- `schema.py` — единственный источник feature profiles;
- `build_windows_dataset.py` — client datasets;
- `build_network_sensor_dataset.py` — sensor dataset;
- `validators.py` — structure, provenance и arithmetic checks.

## Входные данные и выходные данные

Input — normalized/correlated events и execution metadata. Output CSV — runtime artifact в `filin/lab/output/datasets/`.

## Запуск

`python filin/ml/features/validators.py --help`

## Проверки

Client profiles не содержат packet/flow fields, которые недоступны client source. Sensor profile исключает label, execution IDs, marker fields, raw endpoints и Zeek UID.

## Ограничения

Packet/flow признаки требуют независимого sensor source; client events не используются для подмены Zeek observations.

## Связанные документы

[Происхождение данных](../../docs/data-provenance.md), [datasets](../../datasets/README.md).
