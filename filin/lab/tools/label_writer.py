from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_empty_manifest(
    dry_run: bool,
    schedule_mode: str = "grouped",
    gap_seconds: int = 0,
    repeat: int = 1,
) -> dict[str, Any]:
    return {
        "manifest_version": "0.4",
        "lab_name": "Филин v0.2",
        "run_id": f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "timezone": "UTC",
        "dry_run": dry_run,
        "schedule_mode": schedule_mode,
        "gap_seconds": gap_seconds,
        "repeat": repeat,
        "scenario_count": 0,
        "scenarios": [],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return create_empty_manifest(dry_run=True)

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("manifest_version", "0.4")
    data.setdefault("lab_name", "Филин v0.2")
    data.setdefault("run_id", f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")
    data.setdefault("timezone", "UTC")
    data.setdefault("dry_run", True)
    data.setdefault("schedule_mode", "grouped")
    data.setdefault("gap_seconds", 0)
    data.setdefault("repeat", 1)
    data.setdefault("scenarios", [])
    data["scenario_count"] = len(data["scenarios"])
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["scenario_count"] = len(manifest.get("scenarios", []))
    path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def append_scenario_window(
    manifest_path: Path,
    scenario: dict[str, Any],
    run_sequence: int,
    planned_started_at: str,
    planned_finished_at: str,
    dry_run: bool,
    schedule_mode: str,
    gap_seconds: int,
    repeat: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    manifest["dry_run"] = dry_run
    manifest["schedule_mode"] = schedule_mode
    manifest["gap_seconds"] = gap_seconds
    manifest["repeat"] = repeat

    scenario_id = scenario["scenario_id"]
    existing_runs = {
        (item.get("scenario_id"), item.get("run_sequence"))
        for item in manifest.get("scenarios", [])
    }
    if (scenario_id, run_sequence) in existing_runs:
        print(
            "Предупреждение: сценарий с таким run_sequence уже есть в manifest "
            f"и не будет добавлен повторно: {scenario_id} #{run_sequence}"
        )
        save_manifest(manifest_path, manifest)
        return manifest

    manifest["scenarios"].append(
        {
            "run_sequence": run_sequence,
            "scenario_id": scenario_id,
            "type": scenario["type"],
            "label": scenario["expected_label"],
            "source_role": scenario["source_role"],
            "target_role": scenario["target_role"],
            "planned_started_at": planned_started_at,
            "planned_finished_at": planned_finished_at,
            "actual_started_at": None,
            "actual_finished_at": None,
            "duration_seconds": int(scenario["duration_seconds"]),
            "dry_run": dry_run,
            "notes": scenario.get("notes", []),
        }
    )
    save_manifest(manifest_path, manifest)
    return manifest
