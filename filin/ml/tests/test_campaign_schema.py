import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'/'campaigns'))
from campaign_schema import scenario_parameters, validate_campaign
class CampaignSchemaTests(unittest.TestCase):
 def test_variants_are_reproducible(self): self.assertEqual(scenario_parameters(4301,'attack_port_scan'),scenario_parameters(4301,'attack_port_scan'))
 def test_variants_change_with_seed(self): self.assertNotEqual(scenario_parameters(4301,'attack_port_scan'),scenario_parameters(4302,'attack_port_scan'))
