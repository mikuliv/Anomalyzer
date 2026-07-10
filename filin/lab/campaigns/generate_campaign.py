from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from campaign_schema import campaign_metadata, validate_campaign


def generate_campaign(campaign_path: Path, output_dir: Path) -> list[dict]:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    validate_campaign(campaign)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for run in campaign["runs"]:
        entry = {**run, **campaign_metadata(campaign, run), "execution_mode": "docker"}
        (output_dir / f"{run['run_id']}.yaml").write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
        entries.append(entry)
    (output_dir / "campaign_index.json").write_text(json.dumps({"campaign_id": campaign["campaign_id"], "campaign_version": campaign["campaign_version"], "runs": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор manifests кампании независимых executions.")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    entries = generate_campaign(Path(args.campaign), Path(args.output_dir))
    print(f"Сформировано manifests: {len(entries)}")


if __name__ == "__main__":
    main()
