# Лабораторный стенд

## Назначение

Изолированное воспроизводимое выполнение безопасных Docker-сценариев и получение фактических client/sensor observations.

## Что реализовано

Стенд запускает независимые runs, фиксирует execution metadata, создаёт start/end markers, пассивно захватывает трафик и передаёт PCAP в offline Zeek processing.

## Основные файлы

- `campaigns/` — campaign manifests и runners;
- `sensor/` — capture, Zeek, markers и корреляция;
- `tools/` — вспомогательные laboratory scripts.

## Входные и выходные данные

Manifests и безопасные сценарии являются входом. PCAP, Zeek logs, normalized events, statuses и datasets — runtime artifacts в `output/`, не предназначенные для Git.

## Запуск

Перед использованием проверяйте реальный CLI: `python filin/lab/sensor/run_v0_3_sensor_stage.py --help`.

## Проверки

Capture, correlation, aggregation, validator, provenance и split audits подтверждают техническую целостность campaign.

## Ограничения

Стенд не моделирует production network и не предназначен для внешних целей.

## Связанные документы

[Архитектура](../docs/architecture.md), [безопасность](../docs/safety-model.md), [воспроизводимость](../docs/reproducibility.md).
