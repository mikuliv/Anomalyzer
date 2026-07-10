# Филин

«Филин» - прототип платформы для интеллектуального мониторинга сетевого трафика, обнаружения инцидентов информационной безопасности и поддержки работы аналитика.

## Назначение

Платформа предназначена для исследования полного pipeline обнаружения: от лабораторного события и признаков модели до карточки инцидента, MITRE ATT&CK и Sigma-кандидата.

## Отличие от Anomalyzer

Anomalyzer был базовой версией с акцентом на ML-классификацию сетевых аномалий. «Филин» развивает эту идею как платформу: добавляет лабораторный стенд, нормализацию событий, построение признаков, backend API, обработку инцидентов, MITRE mapping, Sigma generation и подготовку к интеграции с SIEM.

## Архитектура

```text
сетевые события -> collector/parser -> feature extraction -> ML detection API -> incident engine -> MITRE mapper -> Sigma generator -> validation lab -> dashboard/SIEM export
```

## Компоненты

- `filin/backend/` - API и обработка инцидентов.
- `filin/lab/` - лабораторный стенд и сценарии.
- `filin/ml/` - подготовка признаков, обучение и оценка моделей.
- `filin/collectors/` - заготовки коллекторов Zeek/Suricata.
- `filin/datasets/` - описание датасетов и учебные примеры.
- `filin/docs/` - архитектура, моделирование, дорожная карта.

## Лабораторный стенд

Стенд нужен для воспроизводимого формирования учебных событий в изолированной среде. Он содержит target-сервисы, traffic-client, control-api, backend, заготовки сенсоров и инструменты подготовки датасета.

Attack-сценарии описываются только как безопасные лабораторные имитации. Они не содержат реальных вредоносных payload-ов и не предназначены для внешних целей.

## Данные и разметка

Manifest не является датасетом. Он описывает план прогона, временные окна и разметку сценариев.

- `execution_events.jsonl` - служебный журнал выполнения сценариев.
- `traffic_events.jsonl` - учебные события активности внутри сценариев.
- `normalized_events.jsonl` - единый формат событий.
- `windows_v0_1.csv` и `flows_v0_1.csv` - датасеты признаков для ML.

`scenario_id`, `run_sequence`, `planned_started_at`, `planned_finished_at`, `label` и `label_type` используются для разметки и анализа, но не должны быть входными признаками модели.

## Признаки для моделей

Каталог признаков находится в `filin/ml/features/feature_catalog.yaml`. Модуль `filin/ml/features/` строит:

- window-level датасет по временным окнам;
- flow-level прототип по группам событий.

В `filin/datasets/examples/` находятся маленькие учебные CSV-примеры. Их можно проверить через `validators.py`, включая базовую арифметику window-level и flow-level признаков.

Сырые события не являются готовыми признаками для модели:

```text
raw events -> normalized events -> feature extraction -> windows.csv / flows.csv -> training
```

## Backend API

Backend на FastAPI содержит базовые endpoint-ы:

- `GET /health`
- `POST /api/v1/predict`
- `POST /api/v1/incidents`
- `GET /api/v1/incidents/{incident_id}`
- `POST /api/v1/sigma/generate`
- `POST /api/v1/rules/validate`

В текущей версии inference является прототипом и подготовлен к дальнейшему подключению реальной модели.

## MITRE ATT&CK и Sigma

MITRE mapping связывает классы событий с осторожными кандидатами техник MITRE ATT&CK. Sigma-модуль генерирует черновик правила. Sigma-кандидат требует проверки на лабораторном стенде и не должен переноситься в SIEM без валидации.

## Быстрый запуск

Backend:

```powershell
cd H:\Anomalyzer\filin\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Лабораторный mock-pipeline:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1

python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --mock --max-runtime-seconds 300

python filin/lab/tools/normalize_events.py --execution-events filin/lab/output/execution_events.jsonl --traffic-events filin/lab/output/traffic_events.jsonl --output filin/lab/output/normalized_events.jsonl

python filin/ml/features/build_windows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/windows_v0_1.csv --window-seconds 60

python filin/ml/features/build_flows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/flows_v0_1.csv

python filin/ml/features/validators.py --csv filin/datasets/examples/windows_v0_1.example.csv --kind windows

python filin/ml/features/validators.py --csv filin/datasets/examples/flows_v0_1.example.csv --kind flows
```

## Ограничения текущей версии

- Mock-режим формирует синтетические лабораторные события без сетевой активности.
- Для обучения итоговых моделей требуется реальный сбор трафика в Docker/VMware-стенде.
- Flow-level датасет v0.1 является прототипом до подключения Zeek/Suricata.
- Постоянное хранилище, web-ui и SIEM export пока не завершены.

## План развития

- Расширить стенд реальными логами Zeek/Suricata.
- Уточнить схему признаков и validation rules.
- Провести корректный train/test split, обучение и сравнение моделей.
- Развить incident engine и проверку Sigma-кандидатов.
- Подготовить демонстрационный сценарий для ВКР/НИР.

## Филин v0.2: реальный Docker traffic collection

Для реального прогона используйте `--docker`: действия выполняются только внутри `filin_lab_net` по фиксированному allowlist. `traffic-client` формирует наблюдения клиента, а не данные независимого сетевого сенсора. `target-ssh-sim` создаёт безопасное TCP-подключение для учебного административного сценария и не является SSH-сервером.

```powershell
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_docker_001 --base-time 2026-07-10T15:00:00Z --gap-seconds 30 --repeat 1 --docker --window-seconds 60 --time-scale 0.05 --random-seed 101 --start-services
```

## Филин v0.2.1 — Docker-to-Docker evaluation

`filin/ml/analysis/` содержит provenance-проверку, анализ PSI и сводный отчёт по нескольким runs. Metadata происхождения (`execution_mode`, `synthetic`, `observation_source`) остаётся исключённой из model features.
# Независимая кампания v0.2.3

Кампания формирует шесть train и три test Docker-runs с разными seeds и параметрами безопасных сценариев.
