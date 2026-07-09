from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


def normalize_zeek_event(event: dict[str, Any], log_type: str) -> dict[str, Any]:
    return {
        "timestamp": event.get("ts"),
        "event_source": "zeek",
        "log_type": log_type,
        "source_ip": event.get("id.orig_h"),
        "source_port": event.get("id.orig_p"),
        "destination_ip": event.get("id.resp_h"),
        "destination_port": event.get("id.resp_p"),
        "protocol": event.get("proto"),
        "duration": event.get("duration"),
        "raw": event,
    }


def normalize_suricata_event(event: dict[str, Any]) -> dict[str, Any]:
    flow = event.get("flow") or {}
    return {
        "timestamp": event.get("timestamp"),
        "event_source": "suricata",
        "log_type": event.get("event_type"),
        "source_ip": event.get("src_ip"),
        "source_port": event.get("src_port"),
        "destination_ip": event.get("dest_ip"),
        "destination_port": event.get("dest_port"),
        "protocol": event.get("proto"),
        "duration": flow.get("age"),
        "raw": event,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Входной JSONL не найден: {path}")
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


def write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalize_execution_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(uuid4()),
        "timestamp": event.get("timestamp"),
        "run_id": event.get("run_id"),
        "run_sequence": event.get("run_sequence"),
        "scenario_id": event.get("scenario_id"),
        "label": event.get("label"),
        "label_type": event.get("type"),
        "source_role": event.get("source_role"),
        "target_role": event.get("target_role"),
        "event_source": "scenario_executor",
        "event_type": event.get("action"),
        "raw_ref": None,
        "features": {
            "execution_status": event.get("status"),
            "requests_sent": (event.get("details") or {}).get("requests_sent"),
            "errors": (event.get("details") or {}).get("errors"),
            "mock": (event.get("details") or {}).get("mock", False),
        },
        "raw": event,
    }


def normalize_execution_events(input_path: Path, output_path: Path) -> int:
    raw_events = read_jsonl(input_path)
    normalized = [normalize_execution_event(event) for event in raw_events]
    write_jsonl(output_path, normalized)
    return len(normalized)


def write_dry_run_sample(output_path: Path) -> None:
    sample = normalize_zeek_event(
        {
            "ts": "2026-07-09T08:00:00Z",
            "id.orig_h": "10.10.0.10",
            "id.orig_p": 41000,
            "id.resp_h": "10.10.0.20",
            "id.resp_p": 80,
            "proto": "tcp",
            "duration": 1.2,
        },
        log_type="conn",
    )
    write_jsonl(output_path, [sample])


def main() -> None:
    parser = argparse.ArgumentParser(description="Нормализация событий лабораторного стенда Филин в JSONL.")
    parser.add_argument("--input", default=None, help="Путь к execution_events.jsonl.")
    parser.add_argument("--output", required=True, help="Путь к выходному normalized_events.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Создать небольшой пример без чтения логов.")
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.dry_run:
        write_dry_run_sample(output_path)
        print(f"Dry-run нормализации записан: {output_path}")
        return

    if not args.input:
        raise ValueError("Для нормализации событий выполнения нужно указать --input.")

    count = normalize_execution_events(Path(args.input), output_path)
    print(f"Нормализованные события записаны: {output_path}")
    print(f"Количество событий: {count}")


if __name__ == "__main__":
    main()
