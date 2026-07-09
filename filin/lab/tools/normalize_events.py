from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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


def write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Заготовка нормализации событий Zeek/Suricata в JSONL.")
    parser.add_argument("--output", required=True, help="Путь к выходному JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Создать небольшой пример без чтения логов.")
    args = parser.parse_args()

    if not args.dry_run:
        raise ValueError("В v0.1 поддерживается только dry-run нормализации.")

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
    write_jsonl(Path(args.output), [sample])
    print(f"Dry-run нормализации записан: {args.output}")


if __name__ == "__main__":
    main()
