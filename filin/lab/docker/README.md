# Docker-стенд Филин v0.1

Docker-стенд v0.1 предназначен для прототипирования лабораторного контура: целевые сервисы, генератор обычного трафика, управляющий сценариями контейнер, заготовки сенсоров, локальное хранилище, просмотр событий и подключение backend Филин.

Стенд не заменяет VMware-стенд v0.2. Версия v0.2 должна использоваться для более реалистичной сети с отдельными виртуальными машинами, полноценной маршрутизацией, выделенными сенсорами и расширенной фиксацией PCAP.

## Сети

- `filin_lab_net` - лабораторный трафик между клиентами и целевыми сервисами.
- `filin_monitor_net` - сенсоры, collector и хранилище.
- `filin_mgmt_net` - backend, Kibana и управляющие компоненты.

## Публикуемые порты

Наружу публикуются только backend и Kibana. Elasticsearch остается доступным внутри Docker-сетей и не публикуется наружу без необходимости.

## Запуск

```powershell
cd filin/lab/docker
docker compose --env-file .env.example -f docker-compose.lab.yml up -d
```

Остановка:

```powershell
docker compose --env-file .env.example -f docker-compose.lab.yml down
```

Перед запуском сценариев нужно убедиться, что все цели входят в allowlist и `FILIN_SCENARIO_MODE` установлен в `dry-run` или другой явно контролируемый режим.

## Запуск лабораторного стенда v0.1

```powershell
cd H:\Anomalyzer\filin\lab\docker
docker compose -f docker-compose.lab.yml up -d --build
docker compose -f docker-compose.lab.yml ps
```

Стенд поднимает внутренние сервисы `target-web`, `target-api`, `control-api` и `traffic-client`. Порты `target-web` и `target-api` могут быть опубликованы на `127.0.0.1` только для отладки. В режиме сбора датасета внешняя публикация этих сервисов не требуется.

## Pipeline сбора

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1

python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --max-runtime-seconds 1800

python filin/lab/tools/normalize_events.py --execution-events filin/lab/output/execution_events.jsonl --traffic-events filin/lab/output/traffic_events.jsonl --output filin/lab/output/normalized_events.jsonl

python filin/lab/tools/dataset_report.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/execution_events.jsonl --traffic-events filin/lab/output/traffic_events.jsonl --normalized filin/lab/output/normalized_events.jsonl --output filin/lab/output/dataset_report.md
```

Для проверки без сетевой активности:

```powershell
python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --mock --max-runtime-seconds 300
```

`execution_events.jsonl` содержит служебные события выполнения, `traffic_events.jsonl` содержит учебные события сетевой активности, `normalized_events.jsonl` содержит единый формат для дальнейшего построения признаков. Mock-режим нужен для проверки pipeline и не заменяет реальный сбор трафика.

## Проверка доступности сервисов

Сначала нужно проверить состояние контейнеров:

```powershell
cd H:\Anomalyzer\filin\lab\docker
docker compose -f docker-compose.lab.yml ps
```

Проверка опубликованных localhost-портов:

```powershell
curl http://127.0.0.1:18080/
curl http://127.0.0.1:18081/
curl http://127.0.0.1:18081/health
```

Если `control-api` не опубликован наружу, его нужно проверять из Docker-сети:

```powershell
docker compose -f docker-compose.lab.yml exec traffic-client python -c "import requests; print(requests.get('http://control-api:8090/health').text)"
```

Автоматическая проверка host-режима:

```powershell
cd H:\Anomalyzer
python filin/lab/tools/check_lab_services.py --mode host
```

Автоматическая проверка Docker DNS-имён выполняется из контейнера, подключенного к `filin_lab_net`:

```powershell
cd H:\Anomalyzer\filin\lab\docker
docker compose -f docker-compose.lab.yml exec traffic-client python /workspace/lab/tools/check_lab_services.py --mode docker
```

В текущей конфигурации `target-api` доступен внутри Docker-сети как `http://target-api:8080`, а `control-api` как `http://control-api:8090`.
