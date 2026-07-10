from __future__ import annotations
import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'features'))
from build_windows_dataset import aggregate_client_window
class Aggregation(unittest.TestCase):
 def test_http_and_tcp(self):
  events=[{'event_type':'http_request','status':'ok','status_code':200,'method':'GET','path':'/','bytes_in':10,'bytes_out':2,'latency_ms':5,'timestamp':'2026-01-01T00:00:00Z','target_host':'target-web','target_port':80},{'event_type':'tcp_connect_check','status':'closed','latency_ms':8,'timestamp':'2026-01-01T00:00:01Z','target_host':'target-web','target_port':22}]
  x=aggregate_client_window(events,2);self.assertEqual(x['event_count'],2);self.assertEqual(x['http_2xx_count'],1);self.assertEqual(x['tcp_closed_count'],1);self.assertEqual(x['bytes_received'],10)
