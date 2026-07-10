import unittest
class CampaignProvenanceTests(unittest.TestCase):
 def test_distinct_hashes_are_required(self): self.assertNotEqual('a','b')
