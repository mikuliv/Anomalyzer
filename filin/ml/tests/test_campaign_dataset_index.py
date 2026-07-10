import unittest
class CampaignDatasetIndexTests(unittest.TestCase):
 def test_train_and_test_are_separate(self): self.assertNotEqual('train','test')
