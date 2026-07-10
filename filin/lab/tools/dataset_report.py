from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"scenarios": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"scenarios": []}


def build_report(
    manifest: dict[str, Any],
    execution_events: list[dict[str, Any]],
    traffic_events: list[dict[str, Any]],
    normalized_events: list[dict[str, Any]],
    service_logs_path: Path | None = None,
) -> str:
    scenarios = manifest.get("scenarios", [])
    labels = Counter(item.get("label", "unknown") for item in scenarios)
    traffic_by_label = Counter(item.get("label", "unknown") for item in traffic_events)
    traffic_by_type = Counter(item.get("event_type", "unknown") for item in traffic_events)
    benign_count = sum(1 for item in scenarios if item.get("type") == "benign")
    attack_count = sum(1 for item in scenarios if item.get("type") == "attack")
    average_traffic = round(len(traffic_events) / len(scenarios), 2) if scenarios else 0
    mode = manifest.get("execution_mode", "mock")
    successful = sum(1 for item in traffic_events if item.get("status") in {"ok", "open"})
    errors = sum(1 for item in traffic_events if item.get("status") in {"error", "closed", "timeout"})
    status_codes = Counter(str(item.get("status_code")) for item in traffic_events if item.get("status_code") is not None)
    latencies = [float(item["latency_ms"]) for item in traffic_events if item.get("latency_ms") is not None]
    logs_available = bool(service_logs_path and service_logs_path.exists() and any(service_logs_path.iterdir()))
    traffic_warning = (
        "Предупреждение: объём событий недостаточен для обучения модели, но подходит для проверки pipeline."
        if len(traffic_events) < 100
        else "Объём traffic events достаточен для проверки расширенного pipeline."
    )

    return "\n".join(
        [
            "# Отчет по лабораторному прогону Филин v0.2",
            "",
            f"- run_id: `{manifest.get('run_id', 'unknown')}`",
            f"- Версия manifest: `{manifest.get('manifest_version', 'unknown')}`",
            f"- Режим расписания: `{manifest.get('schedule_mode', 'unknown')}`",
            f"- Режим выполнения: `{mode}`",
            f"- Количество сценариев: {len(scenarios)}",
            f"- Количество benign-сценариев: {benign_count}",
            f"- Количество attack-сценариев: {attack_count}",
            f"- Список labels: {', '.join(sorted(labels)) if labels else 'нет данных'}",
            f"- Распределение labels: {dict(labels)}",
            f"- Количество execution events: {len(execution_events)}",
            f"- Количество traffic events: {len(traffic_events)}",
            f"- Количество normalized events: {len(normalized_events)}",
            f"- Распределение traffic events по label: {dict(traffic_by_label)}",
            f"- Распределение traffic events по event_type: {dict(traffic_by_type)}",
            f"- Среднее количество traffic events на сценарий: {average_traffic}",
            f"- Успешных действий: {successful}",
            f"- Ошибочных действий: {errors}",
            f"- Количество timeout: {sum(1 for item in traffic_events if item.get('status') == 'timeout')}",
            f"- Количество отказов allowlist: {sum(1 for item in execution_events if 'allowlist' in str(item.get('details', '')).lower())}",
            f"- Распределение status code: {dict(status_codes)}",
            f"- Средняя latency: {round(sum(latencies) / len(latencies), 2) if latencies else 0} мс",
            f"- Наличие service logs: {'да' if logs_available else 'нет'}",
            f"- {traffic_warning}",
            "",
            "## Ограничения",
            "",
            "- Это лабораторный датасет v0.2.",
            "- Mock-режим формирует синтетические лабораторные события и нужен для проверки pipeline." if mode == "mock" else "- События получены в результате реальных действий между контейнерами и наблюдались со стороны traffic-client. Полноценный независимый сетевой мониторинг Zeek/Suricata на этом этапе не выполнялся.",
            "- Для обучения итоговых моделей требуется реальный сбор трафика в Docker/VMware-стенде и последующая проверка качества датасета.",
            "- Низкоинтенсивные attack-сценарии предназначены для разметки и проверки pipeline, а не для имитации реальных атак.",
            "- Сырые PCAP, большие логи и артефакты обучения не хранятся в Git.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание отчета по лабораторному прогону Филин.")
    parser.add_argument("--run-dir", default=None, help="Папка одного laboratory run.")
    parser.add_argument("--manifest", default=None, help="Путь к scenario_manifest.yaml.")
    parser.add_argument("--events", default=None, help="Путь к execution_events.jsonl.")
    parser.add_argument("--traffic-events", default=None, help="Путь к traffic_events.jsonl.")
    parser.add_argument("--normalized", default=None, help="Путь к normalized_events.jsonl.")
    parser.add_argument("--output", default=None, help="Путь к Markdown-отчету.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "scenario_manifest.yaml"
        events_path = run_dir / "execution_events.jsonl"
        traffic_events_path = run_dir / "traffic_events.jsonl"
        normalized_path = run_dir / "normalized_events.jsonl"
        output = Path(args.output) if args.output else run_dir / "dataset_report.md"
    else:
        if not args.manifest or not args.events or not args.normalized or not args.output:
            raise ValueError("Нужно указать --run-dir или явные пути --manifest, --events, --normalized и --output.")
        manifest_path = Path(args.manifest)
        events_path = Path(args.events)
        traffic_events_path = Path(args.traffic_events) if args.traffic_events else None
        normalized_path = Path(args.normalized)
        output = Path(args.output)

    report = build_report(
        manifest=read_manifest(manifest_path),
        execution_events=read_jsonl(events_path),
        traffic_events=read_jsonl(traffic_events_path) if traffic_events_path else [],
        normalized_events=read_jsonl(normalized_path),
        service_logs_path=(run_dir / "service_logs") if args.run_dir else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Отчет записан: {output}")


if __name__ == "__main__":
    main()
