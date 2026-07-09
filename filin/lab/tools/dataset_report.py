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


def build_report(events: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    scenarios = manifest.get("scenarios", [])
    labels = Counter(item.get("label", "unknown") for item in scenarios)
    benign_count = labels.get("benign", 0)
    attack_count = max(0, len(scenarios) - benign_count)

    return "\n".join(
        [
            "# Черновик отчета по датасету",
            "",
            f"- Количество событий: {len(events)}",
            f"- Количество сценариев: {len(scenarios)}",
            f"- Распределение классов: {dict(labels)}",
            f"- Доля benign/attack: {benign_count}/{attack_count}",
            "- Длительность прогона: рассчитывается по первому и последнему временному окну manifest.",
            "- Известные ограничения: Docker-стенд v0.1, лабораторные сервисы, dry-run для части сценариев.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание черновика отчета по лабораторному датасету.")
    parser.add_argument("--events", required=True, help="Путь к JSONL с нормализованными событиями.")
    parser.add_argument("--manifest", required=True, help="Путь к scenario_manifest.yaml.")
    parser.add_argument("--output", required=True, help="Путь к Markdown-отчету.")
    args = parser.parse_args()

    report = build_report(read_jsonl(Path(args.events)), read_manifest(Path(args.manifest)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Отчет записан: {output}")


if __name__ == "__main__":
    main()
