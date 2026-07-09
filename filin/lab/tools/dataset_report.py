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
    normalized_events: list[dict[str, Any]],
) -> str:
    scenarios = manifest.get("scenarios", [])
    labels = Counter(item.get("label", "unknown") for item in scenarios)
    benign_count = sum(1 for item in scenarios if item.get("type") == "benign")
    attack_count = sum(1 for item in scenarios if item.get("type") == "attack")

    return "\n".join(
        [
            "# Отчет по лабораторному прогону Филин v0.1",
            "",
            f"- run_id: `{manifest.get('run_id', 'unknown')}`",
            f"- Версия manifest: `{manifest.get('manifest_version', 'unknown')}`",
            f"- Режим расписания: `{manifest.get('schedule_mode', 'unknown')}`",
            f"- Количество сценариев: {len(scenarios)}",
            f"- Количество benign-сценариев: {benign_count}",
            f"- Количество attack-сценариев: {attack_count}",
            f"- Список labels: {', '.join(sorted(labels)) if labels else 'нет данных'}",
            f"- Распределение labels: {dict(labels)}",
            f"- Количество execution events: {len(execution_events)}",
            f"- Количество normalized events: {len(normalized_events)}",
            "",
            "## Ограничения",
            "",
            "- Это лабораторный датасет v0.1.",
            "- Сценарии выполняются только в изолированной Docker-сети или mock-режиме.",
            "- Низкоинтенсивные attack-сценарии предназначены для разметки и проверки pipeline, а не для имитации реальных атак.",
            "- Сырые PCAP, большие логи и артефакты обучения не хранятся в Git.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание отчета по лабораторному прогону Филин.")
    parser.add_argument("--manifest", required=True, help="Путь к scenario_manifest.yaml.")
    parser.add_argument("--events", required=True, help="Путь к execution_events.jsonl.")
    parser.add_argument("--normalized", required=True, help="Путь к normalized_events.jsonl.")
    parser.add_argument("--output", required=True, help="Путь к Markdown-отчету.")
    args = parser.parse_args()

    report = build_report(
        manifest=read_manifest(Path(args.manifest)),
        execution_events=read_jsonl(Path(args.events)),
        normalized_events=read_jsonl(Path(args.normalized)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Отчет записан: {output}")


if __name__ == "__main__":
    main()
