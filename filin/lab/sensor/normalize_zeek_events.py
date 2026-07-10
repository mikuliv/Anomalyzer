from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from zeek_log_parser import parse_zeek_log
def main():
 p=argparse.ArgumentParser(description='Нормализация событий Zeek из offline-логов.');p.add_argument('--logs-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--run-id',required=True);a=p.parse_args();events=[]
 for path in Path(a.logs_dir).glob('*.log'):
  for raw in parse_zeek_log(path):events.append({'event_id':f"{a.run_id}:{path.name}:{raw.get('uid','')}",'sensor_event_id':raw.get('uid'),'run_id':a.run_id,'event_source':'zeek_sensor','observation_source':'network_sensor','sensor_type':'zeek','sensor_log_type':path.stem,'zeek_uid':raw.get('uid'),'timestamp':raw.get('ts'),'execution_mode':'docker','synthetic':False,'raw':raw})
 Path(a.output).write_text('\n'.join(json.dumps(item,ensure_ascii=False) for item in events)+'\n',encoding='utf-8');print(f'Нормализовано событий: {len(events)}')
if __name__=='__main__':main()
