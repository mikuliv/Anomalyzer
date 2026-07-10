import unittest
class CampaignLoaderTests(unittest.TestCase):
 def test_metadata_is_not_feature(self): self.assertNotIn('run_id',['event_count'])
