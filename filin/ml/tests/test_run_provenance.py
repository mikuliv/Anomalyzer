from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))
from run_provenance import check_run_provenance  # noqa: E402


class ProvenanceTests(unittest.TestCase):
    def test_docker_origin_is_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run_docker_test"
            run.mkdir()
            run_id = "run-test"
            (run / "scenario_manifest.yaml").write_text(yaml.safe_dump({"run_id": run_id, "execution_mode": "docker"}), encoding="utf-8")
            event = {"run_id": run_id, "target_host": "target-web", "execution_mode": "docker", "synthetic": False, "observation_source": "client", "event_source": "traffic_client"}
            for name in ("execution_events.jsonl", "traffic_events.jsonl", "normalized_events.jsonl"):
                (run / name).write_text(json.dumps(event) + "\n", encoding="utf-8")
            dataset = Path(directory) / "windows_v0_1_run_docker_test.csv"
            pd.DataFrame({"run_id": [run_id], "label": ["benign"], "execution_mode": ["docker"], "synthetic": [False], "observation_source": ["client"]}).to_csv(dataset, index=False)
            self.assertTrue(check_run_provenance(run, dataset)["ok"])
