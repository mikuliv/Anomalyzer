from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'filin/ml/analysis'))
from campaign_common import read_jsonl, write_result
import yaml

def audit_campaign_executions(output_root: Path) -> dict:
 status=json.loads((output_root/'campaigns/filin_v0_2_3/campaign_status.json').read_text(encoding='utf-8')); errors=[]; rows=[]
 for run_id,item in status['runs'].items():
  run_dir=output_root/'runs'/run_id; traffic=read_jsonl(run_dir/'traffic_events.jsonl'); manifest=yaml.safe_load((run_dir/'scenario_manifest.yaml').read_text(encoding='utf-8')); actual={item.get('execution_id'):item for item in manifest.get('scenarios',[])}; by={}
  for event in traffic: by.setdefault(event.get('execution_id'),[]).append(event)
  for execution_id, events in by.items():
   first=events[0]; scenario=actual.get(execution_id,{}); duration=0.0
   if scenario.get('actual_started_at') and scenario.get('actual_finished_at'):
    from datetime import datetime
    duration=(datetime.fromisoformat(scenario['actual_finished_at'].replace('Z','+00:00'))-datetime.fromisoformat(scenario['actual_started_at'].replace('Z','+00:00'))).total_seconds()
   label=first.get('label'); rows.append({'run_id':run_id,'role':item['role'],'execution_id':execution_id,'scenario_id':first.get('scenario_id'),'label':label,'parameter_hash':first.get('scenario_parameter_hash'),'seed':first.get('campaign_seed'),'duration_seconds':duration,'traffic_event_count':len(events)})
   minimum={'low_rate_dos':6,'beacon_simulation':8}.get(label)
   if minimum and duration < minimum-0.25: errors.append(f'{execution_id}: недостаточная длительность')
 for role, minimum in [('train',6),('test',3)]:
  for label in ('port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'):
   selected=[row for row in rows if row['role']==role and row['label']==label]
   if len(selected)<minimum or len({row['seed'] for row in selected})<minimum or len({row['parameter_hash'] for row in selected})<minimum: errors.append(f'{role}/{label}: недостаточно независимых executions')
 return {'campaign_execution_valid':not errors,'execution_count':len(rows),'independent_executions':Counter(row['label'] for row in rows),'temporal_durations':[row for row in rows if row['label'] in {'low_rate_dos','beacon_simulation'}],'errors':errors}
def main():
 p=argparse.ArgumentParser(description='Аудит независимых executions кампании.');p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--json-report');a=p.parse_args();r=audit_campaign_executions(Path(a.output_root));write_result(Path(a.json_report) if a.json_report else None,r);print(json.dumps(r,ensure_ascii=False,default=dict));raise SystemExit(0 if r['campaign_execution_valid'] else 1)
if __name__=='__main__':main()
