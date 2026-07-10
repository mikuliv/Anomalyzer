from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description='Сводная проверка sensor-кампании Филин v0.3.');p.add_argument('--campaign',required=True);p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--report-dir',required=True);p.add_argument('--strict',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();root=Path(a.output_root);status=json.loads((root/'campaigns/filin_v0_3/status.json').read_text(encoding='utf-8'));rows=[];errors=[]
 for run,value in status.items():
  dataset=root/'datasets'/f'windows_network_sensor_v0_3_{run}.csv';events=root/'runs'/run/'sensor/normalized_sensor_events.jsonl';assigned=sum(1 for line in events.read_text(encoding='utf-8').splitlines() if '"correlation_status": "assigned"' in line) if events.exists() else 0
  with dataset.open(encoding='utf-8') as source:data=list(csv.DictReader(source)) if dataset.exists() else []
  if value!='success' or len(data)!=13 or assigned==0:errors.append(run)
  rows.append({'run':run,'status':value,'sensor_windows':len(data),'assigned_events':assigned})
 result={'sensor_campaign_completed':not errors,'ready_for_sensor_ml':not errors,'runs':rows,'errors':errors,'ml_training_started':False}
 out=Path(a.report_dir);out.mkdir(parents=True,exist_ok=True);(out/'sensor_stage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'sensor_stage.md').write_text('# Филин v0.3 — sensor stage\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));
 if a.strict and errors:raise SystemExit(1)
if __name__=='__main__':main()
