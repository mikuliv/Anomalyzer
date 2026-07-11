from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DocumentationStatusConsistencyTests(unittest.TestCase):
    def test_completed_versions_are_not_future_work(self):
        roadmap = (ROOT / "filin" / "docs" / "roadmap.md").read_text(encoding="utf-8")
        self.assertIn("v0.3.1 — baseline evaluation", roadmap)
        self.assertIn("v0.3.2 — frozen robustness evaluation", roadmap)
        self.assertNotIn("sensor_ready_for_backend_integration=true", roadmap)

    def test_future_components_are_marked_as_planned(self):
        status = (ROOT / "filin" / "docs" / "status.md").read_text(encoding="utf-8")
        self.assertIn("MITRE ATT&CK mapping | Запланировано", status)
        self.assertIn("Backend model integration | Не начато", status)
