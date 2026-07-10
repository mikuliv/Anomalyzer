from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from validators import validate_dataset


FLOW_METADATA_COLUMNS = ["run_id", "scenario_id", "source_role", "target_role", "event_type", "label", "label_type"]
FLOW_FEATURE_COLUMNS = [
    "total_events",
    "duration_seconds",
    "event_rate",
    "error_count",
    "request_count",
    "bytes_in",
    "bytes_out",
    "unique_dst_ports",
    "unique_paths",
    "failed_login_count",
]


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Ошибка JSON в строке {line_number}: {error}") from error
    return events


def read_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"scenarios": []}


def scenario_labels(manifest: dict[str, Any]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for scenario in manifest.get("scenarios", []):
        scenario_id = str(scenario.get("scenario_id"))
        label_type = str(scenario.get("type", "unknown"))
        label = str(scenario.get("label", "unknown")) if label_type == "attack" else "benign"
        result[scenario_id] = (label, label_type)
    return result


def numeric(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_error(event: dict[str, Any]) -> bool:
    return bool(event.get("error")) or event.get("status") in {"error", "closed"} or numeric(event.get("status_code")) >= 400


def build_flow_rows(manifest: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = scenario_labels(manifest)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event_source") == "scenario_executor":
            continue
        key = (
            str(event.get("source_role") or ""),
            str(event.get("target_role") or ""),
            str(event.get("scenario_id") or ""),
            str(event.get("event_type") or ""),
        )
        grouped[key].append(event)

    rows: list[dict[str, Any]] = []
    for (source_role, target_role, scenario_id, event_type), group in sorted(grouped.items()):
        times = [parse_time(str(event["timestamp"])) for event in group if event.get("timestamp")]
        duration = (max(times) - min(times)).total_seconds() if len(times) > 1 else 1.0
        duration = max(1.0, duration)
        label, label_type = labels.get(scenario_id, ("unknown", "unknown"))
        ports = {event.get("target_port") for event in group if event.get("target_port") is not None}
        paths = {event.get("path") for event in group if event.get("path")}
        row = {
            "run_id": manifest.get("run_id", ""),
            "scenario_id": scenario_id,
            "source_role": source_role,
            "target_role": target_role,
            "event_type": event_type,
            "label": label,
            "label_type": label_type,
            "total_events": float(len(group)),
            "duration_seconds": float(duration),
            "event_rate": float(len(group) / duration),
            "error_count": float(sum(1 for event in group if has_error(event))),
            "request_count": float(sum(1 for event in group if event.get("method") in {"GET", "POST"})),
            "bytes_in": float(sum(numeric(event.get("bytes_in")) for event in group)),
            "bytes_out": float(sum(numeric(event.get("bytes_out")) for event in group)),
            "unique_dst_ports": float(len(ports)),
            "unique_paths": float(len(paths)),
            "failed_login_count": float(
                sum(1 for event in group if event.get("event_type") == "auth_attempt" and event.get("auth_success") is False)
            ),
        }
        rows.append(row)
    return rows


def build_flows_dataset(manifest_path: Path, events_path: Path, output_path: Path) -> int:
    manifest = read_manifest(manifest_path)
    events = read_jsonl(events_path)
    rows = build_flow_rows(manifest, events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FLOW_METADATA_COLUMNS + FLOW_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    validate_dataset(output_path)
    print(f"Flow-level датасет записан: {output_path}")
    print(f"Количество строк: {len(rows)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение flow-level датасета признаков Филин v0.1.")
    parser.add_argument("--run-dir", default=None, help="Папка одного laboratory run.")
    parser.add_argument("--manifest", default=None, help="Путь к scenario_manifest.yaml.")
    parser.add_argument("--events", default=None, help="Путь к normalized_events.jsonl.")
    parser.add_argument("--output", required=True, help="Путь к выходному CSV.")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "scenario_manifest.yaml"
        events_path = run_dir / "normalized_events.jsonl"
    else:
        if not args.manifest or not args.events:
            raise ValueError("Нужно указать --run-dir или оба параметра --manifest и --events.")
        manifest_path = Path(args.manifest)
        events_path = Path(args.events)
    build_flows_dataset(manifest_path, events_path, Path(args.output))


if __name__ == "__main__":
    main()
