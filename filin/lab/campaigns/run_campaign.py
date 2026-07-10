from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT / "filin" / "lab" / "tools", ROOT / "filin" / "ml" / "features"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from build_windows_dataset import build_windows_dataset
from validators import validate_dataset
from run_lab_pipeline import run_pipeline
from scenario_runner import NATURAL_SCENARIO_ORDER
from campaign_schema import build_execution_metadata, campaign_metadata, validate_campaign
from generate_campaign import generate_campaign


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_is_complete(status: dict[str, Any], run_id: str) -> bool:
    item = status.get("runs", {}).get(run_id, {})
    if item.get("status") != "success":
        return False
    return all(Path(path).exists() and sha256(Path(path)) == digest for path, digest in item.get("artifacts", {}).items())


def verify_run(run_dir: Path, core: Path, extended: Path) -> dict[str, Any]:
    manifest = yaml.safe_load((run_dir / "scenario_manifest.yaml").read_text(encoding="utf-8"))
    traffic = read_jsonl(run_dir / "traffic_events.jsonl")
    normalized = read_jsonl(run_dir / "normalized_events.jsonl")
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != 13 or len(traffic) == 0:
        raise ValueError("Run не содержит требуемых сценариев или traffic events.")
    if any(event.get("target_host") not in {"target-web", "target-api", "control-api", "internal-dns", "target-ssh-sim"} for event in traffic):
        raise ValueError("Обнаружена цель вне allowlist.")
    if any(event.get("execution_mode") != "docker" or event.get("synthetic") is not False for event in traffic):
        raise ValueError("Нарушена маркировка происхождения Docker-событий.")
    for profile, dataset in (("client_core_v0_2", core), ("client_extended_v0_2", extended)):
        validate_dataset(dataset, feature_profile=profile)
    return {"scenario_execution_count": len(scenarios), "traffic_event_count": len(traffic), "normalized_event_count": len(normalized), "window_audit": "success", "aggregation_consistency": "success", "core_validator": "success", "extended_validator": "success"}


def run_campaign(campaign_path: Path, output_root: Path, strict: bool = False, resume: bool = False, max_runs: int | None = None, force: bool = False, run_ids: set[str] | None = None) -> dict[str, Any]:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    validate_campaign(campaign)
    campaign_dir = output_root / "campaigns" / "filin_v0_2_3"
    manifests_dir = campaign_dir / "manifests"
    generate_campaign(campaign_path, manifests_dir)
    status_path = campaign_dir / "campaign_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if resume and status_path.exists() else {"campaign_id": campaign["campaign_id"], "runs": {}}
    datasets_dir = output_root / "datasets"
    processed = 0
    for position, run in enumerate(campaign["runs"]):
        run_id = run["run_id"]
        if run_ids and run_id not in run_ids:
            continue
        if resume and not force and run_is_complete(status, run_id):
            continue
        if max_runs is not None and processed >= max_runs:
            break
        run_dir = output_root / "runs" / run_id
        variants = {index: build_execution_metadata(campaign, run, index, scenario_id) for index, scenario_id in enumerate(NATURAL_SCENARIO_ORDER, start=1)}
        args = Namespace(run_dir=str(run_dir), scenarios="filin/lab/scenarios", base_time=f"2026-07-12T{10 + (position % 8):02d}:00:00Z", schedule_mode="natural", gap_seconds=15, repeat=1, mock=False, docker=True, compose_file="filin/lab/docker/docker-compose.lab.yml", compose_project_dir="filin/lab/docker", time_scale=0.05, random_seed=run["random_seed"], start_services=position == 0, rebuild_services=False, stop_services_after_run=False, max_runtime_seconds=240, window_seconds=60)
        try:
            result = run_pipeline(args, campaign_metadata=campaign_metadata(campaign, run), scenario_variants=variants, skip_legacy_dataset=True)
            core = datasets_dir / f"windows_client_core_v0_2_{run_id}.csv"
            extended = datasets_dir / f"windows_client_extended_v0_2_{run_id}.csv"
            build_windows_dataset(result["manifest"], result["normalized_events"], core, 60, "client_core_v0_2", "drop", 1.0, "error")
            build_windows_dataset(result["manifest"], result["normalized_events"], extended, 60, "client_extended_v0_2", "drop", 1.0, "error")
            details = verify_run(run_dir, core, extended)
            paths = [result["manifest"], run_dir / "execution_events.jsonl", result["traffic_events"], result["normalized_events"], core, extended]
            status["runs"][run_id] = {"status": "success", "role": run["role"], "seed": run["random_seed"], **details, "artifacts": {str(path): sha256(path) for path in paths}}
        except Exception as error:
            status["runs"][run_id] = {"status": "failed", "role": run["role"], "seed": run["random_seed"], "error": str(error)}
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            if strict:
                raise
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1
    status_path.with_suffix(".md").write_text("# Статус кампании\n\n" + "\n".join(f"- {key}: {value['status']}" for key, value in status["runs"].items()) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск независимой Docker-кампании Филин v0.2.3.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", default="filin/lab/output")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Ограничить число новых runs одного вызова.")
    parser.add_argument("--force", action="store_true", help="Повторить runs независимо от ранее сохранённого статуса.")
    parser.add_argument("--run-ids", nargs="*", default=None, help="Повторить только перечисленные run IDs.")
    args = parser.parse_args()
    status = run_campaign(Path(args.campaign), Path(args.output_root), args.strict, args.resume, args.max_runs, args.force, set(args.run_ids or []))
    failed = [run_id for run_id, value in status["runs"].items() if value.get("status") != "success"]
    print(f"Успешных runs: {len(status['runs']) - len(failed)}; ошибок: {len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
