# Текущие возможности

## Реализовано

- изолированный Docker laboratory с безопасными локальными сценариями;
- marker-aware executions, passive capture и Docker-managed PCAP storage;
- offline Zeek и профиль `network_sensor_v0_3`;
- train/test campaign v0.3: 9 runs и 117 sensor windows;
- baseline evaluation v0.3.1 и frozen robustness evaluation v0.3.2;
- validators, provenance, split и feature audits.

## Экспериментально подтверждено

В v0.3.1 внешний pooled test `network_sensor_v0_3` дал macro F1 `0.918`, balanced accuracy `0.972` и attack macro recall `1.000`. В v0.3.2 на 12 external robustness-runs — `0.933`, `0.979` и `1.000` соответственно. Это результаты контролируемого стенда, а не гарантия работы в иной среде.

## Частично готово

Исторический backend содержит прототипные endpoint-ы, но модель `network_sensor_v0_3` в него не интегрирована. Не следует трактовать наличие endpoint-ов как готовый production pipeline.

## Запланировано

MITRE ATT&CK mapping, Sigma drafts, SIEM integration, incident representation и analyst interface относятся к будущей концептуальной архитектуре.

Полученные результаты относятся к контролируемому лабораторному стенду и не подтверждают готовность модели к эксплуатации в производственной инфраструктуре.
