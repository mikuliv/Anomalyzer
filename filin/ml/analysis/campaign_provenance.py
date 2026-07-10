from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "filin" / "ml" / "analysis"))
from campaign_common import ALLOWED_TARGETS, read_jsonl, sha256, write_result


def audit_campaign_provenance(output_root: Path) -> dict:
    status = json.loads((output_root / "campaigns" / "filin_v0_2_3" / "campaign_status.json").read_text(encoding="utf-8"))
    errors, hashes, run_ids, seeds = [], set(), set(), set()
    for run_id, item in status.get("runs", {}).items():
        run_ids.add(run_id); seeds.add(item.get("seed"))
        run_dir = output_root / "runs" / run_id
        traffic = read_jsonl(run_dir / "traffic_events.jsonl")
        normalized = read_jsonl(run_dir / "normalized_events.jsonl")
        for event in traffic + normalized:
            if not event.get("campaign_id") or event.get("run_id") != run_id:
                errors.append(f"{run_id}: отсутствует metadata кампании")
                break
        if any(event.get("target_host") and event.get("target_host") not in ALLOWED_TARGETS for event in traffic): errors.append(f"{run_id}: внешняя цель")
        for path in (run_dir / "traffic_events.jsonl", run_dir / "normalized_events.jsonl"):
            value = sha256(path)
            if value in hashes: errors.append(f"{run_id}: совпадающий hash артефакта")
            hashes.add(value)
    if len(run_ids) != 9 or len(seeds) != 9: errors.append("Нарушена уникальность runs или seeds")
    return {"campaign_provenance_valid": not errors, "run_count": len(run_ids), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит происхождения campaign artifacts.")
    parser.add_argument("--output-root", default="filin/lab/output"); parser.add_argument("--json-report")
    args = parser.parse_args(); result = audit_campaign_provenance(Path(args.output_root)); write_result(Path(args.json_report) if args.json_report else None, result); print(json.dumps(result, ensure_ascii=False)); raise SystemExit(0 if result["campaign_provenance_valid"] else 1)
if __name__ == "__main__": main()
