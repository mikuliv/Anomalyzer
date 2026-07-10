import unittest
class GroupedSelectionTests(unittest.TestCase):
 def test_validation_run_is_separate(self): self.assertNotEqual('train_001','train_002')
