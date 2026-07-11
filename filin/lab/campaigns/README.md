# Campaigns

## Назначение

Campaign manifest задаёт независимые laboratory runs, seeds, роли train/test/robustness и безопасные параметры сценариев.

## Что реализовано

Campaign runners сохраняют status и checksums; `--resume` предназначен для продолжения незавершённых attempts без повторения успешных фаз.

## Основные файлы

- `v0_3_zeek_sensor.yaml` — независимая sensor campaign v0.3;
- `v0_3_2_zeek_robustness.yaml` — robustness campaign v0.3.2;
- `run_sensor_campaign.py` — запуск/продолжение sensor campaign.

## Входные данные и выходные данные

Вход — manifest; выход — run statuses и runtime artifacts в `filin/lab/output/`.

## Запуск

Параметры runner: `python filin/lab/campaigns/run_sensor_campaign.py --help`.

## Проверки

Наличие каталога не равно успеху: учитываются statuses, hashes и последующие audits.

## Ограничения

Campaign roles разделяют данные, но не подтверждают production applicability.

## Связанные документы

[Происхождение данных](../../docs/data-provenance.md), [воспроизводимость](../../docs/reproducibility.md).
