from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import yaml
def main():
 p=argparse.ArgumentParser(description='Запуск независимой robustness-кампании Zeek.');p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();c=yaml.safe_load(Path(a.campaign).read_text(encoding='utf-8'));status_path=Path('filin/lab/output/campaigns/filin_v0_3_2/status.json');status=json.loads(status_path.read_text(encoding='utf-8')) if a.resume and status_path.exists() else {}
 for run in c['runs']:
  if status.get(run['run_id'])=='success':continue
  status[run['run_id']]='pending';status_path.parent.mkdir(parents=True,exist_ok=True);status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
 status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()
