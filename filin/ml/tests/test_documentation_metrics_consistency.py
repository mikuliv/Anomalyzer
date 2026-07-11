from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DocumentationMetricsConsistencyTests(unittest.TestCase):
    def test_confirmed_sensor_metrics_are_not_swapped(self):
        text = (ROOT / "filin" / "docs" / "experiments.md").read_text(encoding="utf-8")
        self.assertIn("0.918", text)
        self.assertIn("0.933", text)
        self.assertIn("0.972", text)
        self.assertIn("0.979", text)

    def test_campaign_counts_are_documented(self):
        text = (ROOT / "filin" / "docs" / "experiments.md").read_text(encoding="utf-8")
        self.assertIn("12 robustness-runs", text)
        self.assertIn("156 windows", text)
