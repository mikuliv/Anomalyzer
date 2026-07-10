import unittest
class CampaignSplitAuditTests(unittest.TestCase):
 def test_roles_do_not_overlap(self): self.assertFalse({'train_001'} & {'test_001'})
