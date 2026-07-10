from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))
from feature_drift import analyze, psi  # noqa: E402


class FeatureDriftTests(unittest.TestCase):
    def test_metadata_columns_excluded(self) -> None:
        reference = pd.DataFrame({"run_id": ["a", "a"], "label": ["benign", "port_scan"], "execution_mode": ["docker", "docker"], "value": [0.0, 1.0]})
        result = analyze(reference, reference.copy(), "label", 10, 1e-9, False)
        self.assertEqual([item["feature"] for item in result["features"]], ["value"])

    def test_constant_psi(self) -> None:
        self.assertEqual(psi(pd.Series([0, 0]), pd.Series([0, 0]), 1e-9), (0.0, False))
        value, changed = psi(pd.Series([0, 0]), pd.Series([1, 1]), 1e-9)
        self.assertGreater(value, 0.25)
        self.assertTrue(changed)

    def test_nan_and_zero_are_safe(self) -> None:
        result = analyze(pd.DataFrame({"label": ["benign", "attack"], "value": [0.0, None]}), pd.DataFrame({"label": ["benign", "attack"], "value": [0.0, 2.0]}), "label", 10, 1e-9, False)
        self.assertEqual(len(result["features"]), 1)
