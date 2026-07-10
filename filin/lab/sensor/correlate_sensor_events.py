from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from datetime import datetime
def parse(value):return datetime.fromisoformat(str(value).replace('Z','+00:00')).timestamp()
def main():
 p=argparse.ArgumentParser(description='Корреляция sensor events с фактическими executions.');p.add_argument('--manifest',required=True);p.add_argument('--events',required=True);p.add_argument('--output',required=True);p.add_argument('--tolerance-seconds',type=float,default=1);a=p.parse_args();manifest=yaml.safe_load(Path(a.manifest).read_text(encoding='utf-8'));scenarios=manifest['scenarios'];result=[]
 for line in Path(a.events).read_text(encoding='utf-8').splitlines():
  event=json.loads(line);time=parse(event['timestamp']);matches=[s for s in scenarios if s.get('actual_started_at') and parse(s['actual_started_at'])-a.tolerance_seconds<=time<parse(s['actual_finished_at'])+a.tolerance_seconds]
  if len(matches)==1:
   s=matches[0];event.update({'execution_id':s.get('execution_id'),'scenario_execution_key':f"{manifest['run_id']}:{s['run_sequence']}:{s['scenario_id']}",'scenario_id':s['scenario_id'],'scenario_variant_id':s.get('scenario_variant_id'),'scenario_parameter_hash':s.get('scenario_parameter_hash'),'label':s['label'],'correlation_status':'assigned','correlation_method':'actual_interval'})
  else:event.update({'correlation_status':'background' if not matches else 'ambiguous','correlation_method':'actual_interval'})
  result.append(event)
 Path(a.output).write_text('\n'.join(json.dumps(item,ensure_ascii=False) for item in result)+'\n',encoding='utf-8')
if __name__=='__main__':main()
