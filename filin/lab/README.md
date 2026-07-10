# Лабораторный стенд «Филин»

## Назначение стенда

Лабораторный стенд «Филин» предназначен для воспроизводимого формирования учебных событий сетевой активности, проверки сценариев, нормализации событий, подготовки признаков и будущей оценки моделей обнаружения инцидентов.

Стенд рассчитан только на изолированную лабораторную среду. Он не предназначен для атак на внешние системы.

## Принципы безопасности

- Внешние адреса запрещены.
- Сценарии используют allowlist внутренних сервисов.
- Attack-сценарии являются безопасными лабораторными имитациями.
- Реальные вредоносные payload-ы, эксплойты и команды для внешних целей не используются.
- Сырые PCAP, большие логи и чувствительные данные не коммитятся в Git.

## Состав стенда

- `target-web` - учебный web-сервис.
- `target-api` - учебный API-сервис.
- `control-api` - учебный heartbeat/control endpoint.
- `traffic-client` - контейнер для безопасной генерации лабораторного трафика.
- `scenario_runner.py` - планирование и выполнение сценариев.
- `scenario_executor.py` - безопасное выполнение сценариев и запись событий.
- Zeek/Suricata/Filebeat/Elasticsearch/Kibana - заготовки контура мониторинга.
- FastAPI backend «Филин» - прототип API детекции и инцидентов.

## Сценарии

Сценарии описываются YAML-файлами в `filin/lab/scenarios/`. В них задаются тип активности, метка, источник, цель, длительность, интенсивность и ограничения безопасности.

Benign-сценарии моделируют обычную активность. Attack-сценарии моделируют только безопасные лабораторные признаки подозрительного поведения внутри стенда.

## Manifest-разметка

`scenario_manifest.yaml` фиксирует план прогона:

- `run_id`;
- `run_sequence`;
- `scenario_id`;
- `type`;
- `label`;
- плановое и фактическое время;
- длительность;
- статус выполнения.

Manifest не является датасетом. Он используется для разметки и анализа.

## Режимы расписания

`grouped` - технический режим, где benign-сценарии идут отдельно от attack-сценариев.

`natural` - режим, где attack-сценарии вплетаются в benign-фон. Этот режим используется для более реалистичного лабораторного pipeline.

## Dry-run

Dry-run проверяет сценарии и создает manifest без выполнения сетевых действий.

## Mock execute

Mock execute выполняет manifest без сетевой активности и формирует синтетические лабораторные события. Он нужен для проверки pipeline, структуры данных и построения признаков.

Mock-режим не заменяет реальный сбор трафика и не должен использоваться как финальный датасет для обучения итоговых моделей.

## Docker execute

Docker execute выполняет безопасные обращения к внутренним сервисам Docker-стенда. Все цели должны входить в allowlist. Внешние адреса запрещены.

## Уровни событий

- `execution_events.jsonl` - служебный журнал выполнения сценариев.
- `traffic_events.jsonl` - учебные события сетевой активности внутри сценариев.
- `normalized_events.jsonl` - единый формат событий для дальнейшего построения признаков.
- `windows_v0_1.csv` - window-level датасет признаков.
- `flows_v0_1.csv` - flow-level прототип датасета признаков.

## Нормализация

Нормализация объединяет служебные события и traffic events в общий JSONL-формат. Этот слой еще не является обучающим датасетом: перед ML требуется feature extraction.

## Отчёт по прогону

`dataset_report.md` фиксирует `run_id`, версию manifest, число сценариев, распределение labels, количество execution events, traffic events и normalized events.

## Типовой порядок запуска

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1

python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --mock --max-runtime-seconds 300

python filin/lab/tools/normalize_events.py --execution-events filin/lab/output/execution_events.jsonl --traffic-events filin/lab/output/traffic_events.jsonl --output filin/lab/output/normalized_events.jsonl

python filin/lab/tools/dataset_report.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/execution_events.jsonl --traffic-events filin/lab/output/traffic_events.jsonl --normalized filin/lab/output/normalized_events.jsonl --output filin/lab/output/dataset_report.md

Get-Content filin/lab/output/dataset_report.md -Encoding UTF8
```

Построение признаков:

```powershell
python filin/ml/features/build_windows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/windows_v0_1.csv --window-seconds 60

python filin/ml/features/build_flows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/flows_v0_1.csv
```

## Оценка по разным laboratory runs

Для более честной проверки baseline-моделей нужно обучать модель на одном прогоне стенда, а оценивать на другом. Это снижает риск того, что модель выучит особенности одного конкретного `run_id` или расписания сценариев.

## Несколько laboratory runs

Один laboratory run не должен использоваться одновременно как единственный источник для оценки переносимости модели. Для более строгой проверки рекомендуется обучать модель на одном прогоне, а оценивать на другом.

Пример workflow:

```powershell
# Прогон 1
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_001 --base-time 2026-07-09T13:00:00Z --gap-seconds 30 --repeat 1 --mock --window-seconds 60

# Прогон 2
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_002 --base-time 2026-07-10T13:00:00Z --gap-seconds 45 --repeat 1 --mock --window-seconds 60

# Обучение на run_001 и оценка на run_002
python filin/ml/training/run_external_experiment.py --train-run run_001 --test-run run_002 --target label
```

Даже оценка `run_001 -> run_002` остаётся лабораторной, если оба прогона сформированы в mock-режиме. Такой эксперимент проверяет переносимость между разными прогонами генератора, но не подтверждает качество модели на реальном сетевом трафике.

Артефакты в `filin/lab/output/` не коммитятся.

## Ограничения v0.1

Mock-режим формирует синтетические лабораторные события без сетевой активности. Он нужен для проверки pipeline. Для обучения итоговых моделей требуется реальный сбор трафика в Docker/VMware-стенде.

Docker-стенд v0.1 является прототипом и не заменяет отдельный стенд с виртуальными машинами, выделенными сенсорами и расширенной фиксацией PCAP.

## Филин v0.2: реальный Docker traffic collection

Режим `docker` выполняет сценарии в `traffic-client` и сохраняет реальные результаты HTTP, DNS-разрешений и TCP connect-проверок в `traffic_events.jsonl`. Внешние адреса, произвольные URL и неразрешённые цели блокируются allowlist. `execution_events.jsonl` остаётся служебным журналом; `normalized_events.jsonl` сохраняет `execution_mode`, `synthetic` и `observation_source` только как metadata.

```powershell
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_docker_001 --base-time 2026-07-10T15:00:00Z --gap-seconds 30 --repeat 1 --docker --window-seconds 60 --time-scale 0.05 --random-seed 101
```

## Филин v0.2.1 — Docker-to-Docker evaluation

Для сравнения Docker runs меняются `base-time`, `gap-seconds`, `time-scale` и `random-seed`. Каждый run сохраняет собственные manifest, JSONL и datasets; перенос CSV между runs не допускается.
