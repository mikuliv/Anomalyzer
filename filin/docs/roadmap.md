# Roadmap

## Завершённые этапы

- v0.2.2 — source-aware feature profiles и actual execution windows.
- v0.2.3 — независимые laboratory executions с train/test separation.
- v0.2.4 — baseline evaluation client profiles.
- v0.3 — независимый Zeek sensor pipeline и marker-aware correlation.
- v0.3.1 — baseline evaluation `network_sensor_v0_3`.
- v0.3.2 — frozen robustness evaluation.

## Текущий документационный этап

Нормализация и актуализация документации проекта «Филин»: единые термины, проверяемые ссылки, разделение реализованного и запланированного, фиксация ограничений.

## Ближайший технический этап

**Филин v0.3.3 — расширение проверки на изменённой сетевой среде и более разнообразном фоновом трафике.** Этап не начат. Его цель — расширить независимую внешнюю проверку, не подменяя её дополнительными связанными окнами.

## Среднесрочные этапы

- дополнительная validation на иной инфраструктуре и сервисной топологии;
- более разнообразные безопасные background patterns;
- воспроизводимая процедура deployment validation после накопления данных;
- отдельная оценка наблюдаемости encrypted traffic metadata.

## Долгосрочная концепция

Incident representation, MITRE ATT&CK mapping, Sigma drafts, test bench validation, SIEM integration и analyst interface остаются будущими направлениями. Они не реализованы в текущем pipeline.
