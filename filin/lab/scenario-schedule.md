# Пример расписания 4-часового прогона

Расписание предназначено для изолированного стенда Филин v0.1. Временные окна нужны для последующей разметки событий и проверки качества датасета.

| Время | Окно | Сценарий | Ожидаемая метка |
| --- | --- | --- | --- |
| 00:00-00:30 | Фоновая активность | Обычные HTTP-запросы, DNS, API | `benign` |
| 00:30-00:35 | Контролируемый сценарий | Сканирование портов по allowlist | `port_scan` |
| 00:35-01:10 | Фоновая активность | Web, API, скачивание небольших файлов | `benign` |
| 01:10-01:15 | Контролируемый сценарий | Ошибки авторизации на тестовой учетной записи | `auth_failures` |
| 01:15-01:50 | Фоновая активность | Обычные запросы и DNS | `benign` |
| 01:50-01:55 | Контролируемый сценарий | Web-probe без вредоносных payload-ов | `web_probe` |
| 01:55-02:30 | Фоновая активность | API и скачивание небольших файлов | `benign` |
| 02:30-02:32 | Контролируемый сценарий | Low-rate traffic spike внутри лаборатории | `low_rate_dos` |
| 02:32-03:10 | Фоновая активность | HTTP, DNS, API | `benign` |
| 03:10-03:20 | Контролируемый сценарий | Beacon-симуляция через внутренний control-api | `beacon_simulation` |
| 03:20-04:00 | Контрольное окно | Обычная активность без attack-сценариев | `benign` |

## Правила прогона

1. Перед стартом проверить allowlist целей.
2. Запускать сценарии только через `scenario_runner.py` в dry-run или явно контролируемом режиме.
3. Фиксировать начало и конец каждого окна в `scenario_manifest.yaml`.
4. Не смешивать два attack-сценария в одном временном окне.
5. После прогона сформировать нормализованный JSONL и черновик отчета по датасету.

## Проверка dry-run в Windows PowerShell

Для проверки расписания без генерации сетевого трафика можно запустить все YAML-сценарии из папки:

```powershell
cd H:\Anomalyzer

python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z

Get-Content filin/lab/output/scenario_manifest.yaml -Encoding UTF8
```

В этом режиме `scenario_runner.py` рекурсивно читает сценарии, сначала проверяет benign-сценарии, затем attack-сценарии, рассчитывает плановые временные окна и сохраняет manifest в UTF-8.

Если кириллица в консоли отображается некорректно, можно выполнить:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## Режимы расписания

`grouped` используется для технической проверки манифестов: все benign-сценарии идут одним блоком, затем все attack-сценарии.

`natural` используется для будущего сбора датасета: обычная активность чередуется с контролируемыми attack-сценариями, а повторяющиеся benign-сценарии создают фон между тестовыми окнами.

Пример natural-запуска с паузой 30 секунд между окнами:

```powershell
python filin/lab/tools/scenario_runner.py --scenarios filin/lab/scenarios --manifest filin/lab/output/scenario_manifest.yaml --dry-run --reset-manifest --base-time 2026-07-09T13:00:00Z --schedule-mode natural --gap-seconds 30 --repeat 1

Get-Content filin/lab/output/scenario_manifest.yaml -Encoding UTF8
```

В manifest v0.3 сохраняются `schedule_mode`, `gap_seconds`, `repeat` и `run_sequence`. Повтор одного `scenario_id` допустим, если отличается `run_sequence`.

## Выполнение по manifest

После проверки расписания можно выполнить сценарии последовательно без ожидания планового времени:

```powershell
python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --max-runtime-seconds 1800
```

Для проверки без Docker-сервисов используется mock-режим:

```powershell
python filin/lab/tools/scenario_runner.py --manifest filin/lab/output/scenario_manifest.yaml --execute --allow-dry-run-manifest --mock --max-runtime-seconds 300
```

Если указан `--respect-schedule`, runner будет ждать наступления `planned_started_at`. По умолчанию сценарии выполняются последовательно без ожидания.

После выполнения появляются три уровня файлов:

- `execution_events.jsonl` - служебный журнал выполнения сценариев;
- `traffic_events.jsonl` - учебные события сетевой активности внутри сценариев;
- `normalized_events.jsonl` - единый формат событий для дальнейшего построения признаков.
