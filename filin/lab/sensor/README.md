# Сетевой сенсор Zeek

## Назначение

Независимое наблюдение фактически захваченного Docker-трафика и его преобразование в `network_sensor_v0_3`.

## Что реализовано

Capture-sidecar пассивно наблюдает namespace `traffic-client`; PCAP хранится в Docker named volume. Offline Zeek создаёт logs, parser и normalizer формируют sensor events, а markers задают интервалы корреляции.

## Основные файлы

- `capture_preflight.py` — проверка capture и marker protocol;
- `zeek_log_parser.py`, `normalize_zeek_events.py` — обработка Zeek logs;
- `correlate_sensor_events.py` — marker-aware correlation;
- `run_v0_3_sensor_stage.py` — stage runner.

## Входные данные и выходные данные

Первичный источник — PCAP. Zeek events не создаются из `traffic_events.jsonl`, client CSV или labels. Выходные logs, events и CSV — runtime artifacts.

## Запуск

`python filin/lab/sensor/run_v0_3_sensor_stage.py --help`

## Проверки

Markers реально проходят сеть, но исключаются из aggregation. Correlation не использует labels и не расширяет tolerance до минут.

## Ограничения

Наблюдаемость определяется топологией и видимостью Zeek logs.

## Связанные документы

[Архитектура](../../docs/architecture.md), [provenance](../../docs/data-provenance.md), [ограничения](../../docs/limitations.md).
