from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DocumentationStructureTests(unittest.TestCase):
    def test_required_documents_exist(self):
        required = ("index.md", "architecture.md", "experiments.md", "data-provenance.md", "limitations.md")
        for name in required:
            self.assertTrue((ROOT / "filin" / "docs" / name).is_file(), name)

    def test_main_readmes_link_to_index(self):
        for name in (ROOT / "README.md", ROOT / "filin" / "README.md"):
            self.assertIn("docs/index.md", name.read_text(encoding="utf-8"))
