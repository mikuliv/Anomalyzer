# Filin

Filin - прототип платформы для интеллектуального мониторинга сетевого трафика и поддержки расследования инцидентов информационной безопасности.

## Отличие от Anomalyzer

Anomalyzer был базовой версией с акцентом на ML-классификацию сетевых аномалий. Filin развивает эту идею как платформу: добавляет нормализацию событий, API детекции, карточки инцидентов, сопоставление с MITRE ATT&CK, генерацию Sigma-кандидатов и лабораторную проверку правил.

## Текущий статус

Статус проекта: `prototype`. Реальный inference, постоянное хранилище, web-ui и интеграции с SIEM будут добавляться поэтапно.

## Архитектура

Поток обработки:

`network events -> collector/parser -> feature extractor -> ML detection API -> incident engine -> MITRE mapper -> Sigma generator -> validation lab -> dashboard/SIEM export`

## Быстрый запуск backend

```bash
cd filin/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Для Windows PowerShell:

```powershell
cd filin/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Примеры запросов

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @../examples/sample_event.json
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/incidents \
  -H "Content-Type: application/json" \
  -d @../examples/sample_event.json
```

## Roadmap

- 1 год: рефакторинг исходной идеи, архитектура платформы, API детекции.
- 2 год: incident engine, MITRE mapping, Sigma prototype, лабораторный стенд.
- 3 год: эксперименты с моделями, оценка FP/FN, оформление ВКР и демонстрационный сценарий.
