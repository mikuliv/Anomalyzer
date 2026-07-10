import unittest
class CampaignGenerationTests(unittest.TestCase):
 def test_execution_identifier_is_unique(self): self.assertNotEqual('run:1:attack_port_scan','run:2:attack_port_scan')
