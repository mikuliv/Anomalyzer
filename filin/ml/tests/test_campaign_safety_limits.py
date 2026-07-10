import unittest
class CampaignSafetyTests(unittest.TestCase):
 def test_external_target_is_not_allowed(self): self.assertNotIn('example.org',{'target-web','target-api','control-api'})
