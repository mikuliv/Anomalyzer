# Docker-стенд Филин v0.1

## Назначение Docker-стенда

Docker-стенд v0.1 нужен для проверки лабораторного pipeline: целевые сервисы, безопасный генератор трафика, выполнение сценариев, backend, заготовки сенсоров, локальное хранилище и просмотр событий.

Стенд предназначен только для изолированной лабораторной среды и не заменяет будущий VMware-стенд с отдельными виртуальными машинами.

## Сервисы

- `target-web` - учебный nginx-сервис со статическими страницами и маленькими файлами.
- `target-api` - учебный FastAPI-сервис для API-запросов и auth failures.
- `control-api` - учебный heartbeat/control endpoint для проверки периодичности обращений.
- `traffic-client` - контейнер для запуска безопасных проверок и сценариев внутри Docker-сети.
- `filin-backend` - backend платформы «Филин».
- `zeek`, `suricata`, `filebeat`, `elasticsearch`, `kibana` - заготовки мониторингового контура.

## Сети

- `filin_lab_net` - лабораторный трафик между клиентами и целевыми сервисами.
- `filin_monitor_net` - сенсоры, collector и хранилище.
- `filin_mgmt_net` - backend, Kibana и управляющие компоненты.

## Порты с хоста

```text
target-web    -> http://127.0.0.1:18080/
target-api    -> http://127.0.0.1:18081/health
filin-backend -> http://127.0.0.1:8000/health
```

Elasticsearch не публикуется наружу без необходимости. `control-api` по умолчанию проверяется из Docker-сети.

## Внутренние порты Docker-сети

```text
target-web    -> http://target-web/
target-api    -> http://target-api:8080/health
control-api   -> http://control-api:8090/health
filin-backend -> http://filin-backend:8000/health
```

## Проверка доступности сервисов

Проверка с хоста:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/check_lab_services.py --mode host
```

Проверка Docker DNS-имён через `traffic-client`:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/check_lab_services.py --mode compose-exec
```

Ручная проверка внутри Docker-сети:

```powershell
cd H:\Anomalyzer\filin\lab\docker

docker compose -f docker-compose.lab.yml exec traffic-client python -c "import requests; print(requests.get('http://target-web/').status_code)"

docker compose -f docker-compose.lab.yml exec traffic-client python -c "import requests; print(requests.get('http://target-api:8080/health').text)"

docker compose -f docker-compose.lab.yml exec traffic-client python -c "import requests; print(requests.get('http://control-api:8090/health').text)"
```

`--mode docker` должен выполняться внутри контейнера, подключенного к сети стенда. При запуске напрямую с хоста Docker DNS-имена не будут разрешаться.

## Запуск

```powershell
cd H:\Anomalyzer\filin\lab\docker

docker compose -f docker-compose.lab.yml up -d --build
docker compose -f docker-compose.lab.yml ps
```

## Остановка

```powershell
cd H:\Anomalyzer\filin\lab\docker

docker compose -f docker-compose.lab.yml down
```

## Диагностика

```powershell
docker compose -f docker-compose.lab.yml ps
docker compose -f docker-compose.lab.yml logs target-web
docker compose -f docker-compose.lab.yml logs target-api
docker compose -f docker-compose.lab.yml logs control-api
```

Если `target-api` доступен с хоста, корневой endpoint должен возвращать описание лабораторного сервиса:

```powershell
curl http://127.0.0.1:18081/
curl http://127.0.0.1:18081/health
```

## Ограничения

Docker-стенд v0.2 предназначен для безопасного сбора учебных client observations. Для итогового обучения моделей требуется независимый сбор трафика в Docker/VMware-стенде, проверка качества датасета и контроль отсутствия чувствительных данных.

## Филин v0.2: реальный Docker traffic collection

`traffic-client` подключён только к внутренней `filin_lab_net`, не публикует порты, не использует Docker socket, host network или privileged mode. Разрешены только `target-web`, `target-api`, `control-api`, `target-ssh-sim` и внутренние DNS-имена. `target-ssh-sim:2222` выдаёт тестовый banner и сразу закрывает соединение: это не SSH-сервер, shell и аутентификация отсутствуют.

Полный запуск и сбор диагностических логов:

```powershell
cd H:\Anomalyzer
python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/run_docker_001 --base-time 2026-07-10T15:00:00Z --gap-seconds 30 --repeat 1 --docker --window-seconds 60 --time-scale 0.05 --random-seed 101 --start-services
```

Логи сервисов сохраняются в `<run-dir>/service_logs/` и не добавляются в Git.
