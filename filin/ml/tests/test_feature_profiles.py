from __future__ import annotations
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'features'))
from schema import CLIENT_CORE_V0_2,CLIENT_EXTENDED_V0_2,PACKET_FEATURES,get_feature_profile,get_metadata_columns
class FeatureProfiles(unittest.TestCase):
 def test_profiles(self):
  self.assertEqual(get_feature_profile('legacy_v0_1'),[]);self.assertTrue(set(CLIENT_CORE_V0_2)<=set(CLIENT_EXTENDED_V0_2));self.assertFalse(set(CLIENT_EXTENDED_V0_2)&PACKET_FEATURES);self.assertFalse(set(CLIENT_CORE_V0_2)&set(get_metadata_columns()))
 def test_network_sensor_is_planned(self):
  with self.assertRaises(ValueError):get_feature_profile('network_sensor_v0_3')
