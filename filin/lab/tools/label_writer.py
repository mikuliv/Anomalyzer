from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "manifest_version": "0.1",
            "lab_name": "Филин v0.1",
            "run_id": f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            "timezone": "UTC",
            "scenarios": [],
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("scenarios", [])
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def append_scenario_window(
    manifest_path: Path,
    scenario: dict[str, Any],
    started_at: str,
    finished_at: str,
    dry_run: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    manifest["scenarios"].append(
        {
            "scenario_id": scenario["scenario_id"],
            "type": scenario["type"],
            "label": scenario["expected_label"],
            "source_role": scenario["source_role"],
            "target_role": scenario["target_role"],
            "started_at": started_at,
            "finished_at": finished_at,
            "dry_run": dry_run,
            "notes": scenario.get("notes", []),
        }
    )
    save_manifest(manifest_path, manifest)
    return manifest
