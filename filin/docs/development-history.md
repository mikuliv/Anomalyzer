# История разработки

| Версия | Цель и результат | Данные и проверки | Ограничение |
| --- | --- | --- | --- |
| v0.2.2 | Source-aware profiles и окна по actual execution intervals. | Docker client observations, validators и aggregation audits. | Недостаточно независимых attack executions для ML. |
| v0.2.3 | Независимые train/test Docker-runs. | Campaign manifests, hashes и split discipline. | Client observations не являются независимым sensor source. |
| v0.2.4 | Baseline client profiles. | Независимые runs и external evaluation. | Client profile не перенёсся надёжно на test. |
| v0.3 | Независимый Zeek sensor pipeline. | 9 runs, 117 windows, PCAP→Zeek pipeline, marker correlation. | Лабораторная топология и ограниченный support. |
| v0.3.1 | Сравнение client и `network_sensor_v0_3`. | 6 train, 3 test runs; LORO и external test. | Результаты не подтверждают production readiness. |
| v0.3.2 | Внешняя robustness evaluation frozen baseline. | 12 runs, 156 windows, topology/background/temporal/combined shifts. | Идентичность метрик между runs требует осторожной интерпретации. |

Связанные подтверждённые коммиты: `509b4f0` (passive capture), `e2a5dad` (campaign), `3ea23f9` (audits), `0428177` (baseline) и `9f40b58` (robustness evaluation).
