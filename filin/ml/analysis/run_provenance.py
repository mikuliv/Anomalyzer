from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = REPO_ROOT / "filin" / "ml" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import calculate_file_sha256  # noqa: E402


ALLOWED_TARGETS = {"target-web", "target-api", "control-api", "target-ssh-sim", "internal-dns"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_run_provenance(run_dir: Path, windows_dataset: Path) -> dict[str, Any]:
    required = {
        "manifest": run_dir / "scenario_manifest.yaml",
        "execution_events": run_dir / "execution_events.jsonl",
        "traffic_events": run_dir / "traffic_events.jsonl",
        "normalized_events": run_dir / "normalized_events.jsonl",
        "windows_dataset": windows_dataset,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError("Отсутствуют обязательные артефакты run: " + ", ".join(missing))
    manifest = yaml.safe_load(required["manifest"].read_text(encoding="utf-8")) or {}
    traffic = read_jsonl(required["traffic_events"])
    normalized = read_jsonl(required["normalized_events"])
    dataset = pd.read_csv(windows_dataset, encoding="utf-8")
    run_id = str(manifest.get("run_id", ""))
    external_targets = sorted({str(event.get("target_host")) for event in traffic if event.get("target_host") not in ALLOWED_TARGETS})
    dataset_run_ids = {str(value) for value in dataset.get("run_id", pd.Series(dtype=str)).dropna().unique()}
    checks = {
        "execution_mode_docker": manifest.get("execution_mode") == "docker",
        "traffic_marking": bool(traffic) and all(event.get("execution_mode") == "docker" and event.get("synthetic") is False and event.get("observation_source") == "client" for event in traffic),
        "normalized_marking": bool(normalized) and all(event.get("execution_mode") == "docker" and event.get("synthetic") is False and event.get("observation_source") == "client" for event in normalized if event.get("event_source") == "traffic_client"),
        "run_id_consistent": bool(run_id) and {str(event.get("run_id")) for event in traffic} == {run_id} and dataset_run_ids == {run_id},
        "no_external_targets": not external_targets,
        "dataset_matches_run_name": windows_dataset.name == f"windows_v0_1_{run_dir.name}.csv",
    }
    allowlist_rejections = sum("allowlist" in str(event.get("details", "")).lower() for event in read_jsonl(required["execution_events"]))
    return {
        "run_dir": str(run_dir), "run_id": run_id, "windows_dataset": str(windows_dataset),
        "dataset_sha256": calculate_file_sha256(windows_dataset), "traffic_events": len(traffic),
        "normalized_events": len(normalized), "allowlist_rejections": allowlist_rejections,
        "external_targets": external_targets, "checks": checks, "ok": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка происхождения датасета одного Docker-прогона.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--windows-dataset", required=True)
    parser.add_argument("--output", default=None, help="Необязательный путь к JSON-результату.")
    args = parser.parse_args()
    result = check_run_provenance(Path(args.run_dir), Path(args.windows_dataset))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["ok"]:
        raise SystemExit("Проверка происхождения завершилась с ошибкой.")


if __name__ == "__main__":
    main()
