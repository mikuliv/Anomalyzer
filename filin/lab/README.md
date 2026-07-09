# Лабораторный стенд Филин

Лабораторный стенд Филин v0.1 предназначен для воспроизводимого сбора собственного датасета сетевых событий. Он используется для генерации обычного фонового трафика, запуска контролируемых лабораторных сценариев, сбора логов сенсоров, нормализации событий, разметки временных окон и подготовки данных для будущего обучения моделей.

Стенд рассчитан только на изолированную лабораторную сеть. Сценарии не должны обращаться к внешним адресам и должны использовать allowlist внутренних сервисов.

## Версии стенда

Docker-стенд v0.1 нужен для прототипирования структуры, проверки сценариев, формата разметки и инструментов подготовки датасета. Он не заменяет VMware-стенд v0.2, где позже можно будет развернуть более реалистичную сеть с отдельными виртуальными машинами, полноценными сенсорами и управляемыми узлами.

## Общий поток

`сценарий -> трафик -> сенсор -> лог -> нормализация -> разметка -> датасет`

1. YAML-сценарий задает тип активности, источник, цель, длительность, интенсивность и ограничения.
2. Генератор выполняет только безопасные обращения к внутренним лабораторным сервисам.
3. Zeek или Suricata фиксируют сетевые события.
4. Логи приводятся к единому JSONL-формату.
5. Manifest связывает временные окна сценариев с метками.
6. Dataset report описывает состав, ограничения и распределение классов.

## Минимальный состав

- целевые сервисы: web и API;
- генератор обычного трафика;
- контейнер контролируемых лабораторных сценариев;
- управляющий scenario-runner;
- сенсоры: Zeek и Suricata или их заготовки;
- collector: Filebeat или Logstash;
- хранилище: Elasticsearch;
- просмотр событий: Kibana;
- backend: существующий FastAPI backend Филин.

## Проверка dry-run в Windows PowerShell

Команда для проверки всех сценариев из корня репозитория:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z

Get-Content filin/lab/output/scenario_manifest.yaml -Encoding UTF8
```

Manifest сохраняется в UTF-8. Если кириллица отображается некорректно, файл нужно читать с явным указанием `-Encoding UTF8`.

При необходимости можно переключить консоль PowerShell на UTF-8:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## Режимы расписания

`grouped` - технический режим проверки сценариев. В этом режиме сначала выполняются benign-сценарии, затем attack-сценарии, внутри групп используется сортировка по имени файла.

`natural` - режим для будущего сбора датасета. В этом режиме контролируемые attack-сценарии вплетаются в benign-фон, чтобы поток событий был ближе к обычной сетевой активности лаборатории.

Пример запуска natural-режима:

```powershell
python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1

Get-Content filin/lab/output/scenario_manifest.yaml -Encoding UTF8
```

Параметр `--gap-seconds` добавляет паузы между плановыми окнами. Параметр `--repeat` задает количество повторов natural-последовательности. Для каждого запуска сценария в manifest записывается уникальный `run_sequence`.

## Запуск лабораторного стенда v0.1

Запуск Docker-стенда:

```powershell
cd H:\Anomalyzer\filin\lab\docker
docker compose -f docker-compose.lab.yml up -d --build
docker compose -f docker-compose.lab.yml ps
```

Создание natural manifest:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1
```

Выполнение сценариев по manifest:

```powershell
python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --max-runtime-seconds 1800
```

Если Docker-сервисы недоступны, можно проверить pipeline без сетевой активности:

```powershell
python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --mock --max-runtime-seconds 300
```

Нормализация событий:

```powershell
python filin/lab/tools/normalize_events.py --input filin/lab/output/execution_events.jsonl --output filin/lab/output/normalized_events.jsonl
```

Создание отчета:

```powershell
python filin/lab/tools/dataset_report.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/execution_events.jsonl --normalized filin/lab/output/normalized_events.jsonl --output filin/lab/output/dataset_report.md
```

Чтение manifest:

```powershell
Get-Content filin/lab/output/scenario_manifest.yaml -Encoding UTF8
```

В режиме сбора датасета внешняя публикация `target-web` и `target-api` не требуется. Их localhost-порты в Docker compose предназначены только для отладки.
