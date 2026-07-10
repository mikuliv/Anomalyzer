from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
sys.modules.pop("report_writer", None)
from train_baselines import train_baselines  # noqa: E402


def write_dataset(path: Path, run_id: str) -> None:
    fields = ["run_id", "run_sequence", "scenario_id", "window_start", "window_end", "label", "label_type", "execution_mode", "synthetic", "observation_source", "feature_a", "feature_b"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(60):
            attack = index % 2 == 0
            writer.writerow({"run_id": run_id, "run_sequence": 1, "scenario_id": "scenario", "window_start": "2026-01-01", "window_end": "2026-01-01", "label": "port_scan" if attack else "benign", "label_type": "attack" if attack else "benign", "execution_mode": "docker", "synthetic": False, "observation_source": "client", "feature_a": index, "feature_b": 10 if attack else 1})


class MultiRunTrainingTests(unittest.TestCase):
    def test_same_train_and_test_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            write_dataset(path, "run-a")
            with self.assertRaisesRegex(ValueError, "нельзя включать"):
                train_baselines(path, [], path, "label", Path(directory) / "artifacts", Path(directory) / "report.md", .3, 42, 2)

    def test_additional_train_dataset_is_combined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, test = root / "first.csv", root / "second.csv", root / "test.csv"
            write_dataset(first, "run-a")
            write_dataset(second, "run-b")
            write_dataset(test, "run-c")
            result = train_baselines(first, [second], test, "label", root / "artifacts", root / "report.md", .3, 42, 2)
            self.assertEqual(result["best_model"], "LogisticRegression")
