from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"features"))
from build_windows_dataset import read_jsonl,read_manifest,build_scenario_execution_windows
def main():
 p=argparse.ArgumentParser(description="Симуляция размеров окон Docker scenario executions.");p.add_argument("--run-dir",required=True);p.add_argument("--sizes",nargs="+",type=int,required=True);p.add_argument("--report",required=True);p.add_argument("--json-report",required=True);a=p.parse_args();r=Path(a.run_dir);m=read_manifest(r/'scenario_manifest.yaml');e=[x for x in read_jsonl(r/'normalized_events.jsonl') if x.get('event_source')=='traffic_client'];out=[]
 for size in a.sizes:
  rows,_=build_scenario_execution_windows(m,e,size,'client_core_v0_2','drop',1.,'error'); counts=Counter(x['label'] for x in rows); values=[x['window_event_count'] for x in rows];out.append({'window_size_seconds':size,'all_generated_windows':len(rows),'active_windows':len(rows),'empty_windows':0,'support':dict(counts),'minimum_events_per_active_window':min(values),'median_events_per_active_window':sorted(values)[len(values)//2],'maximum_events_per_active_window':max(values),'classes_with_support_below_5':sorted(k for k,v in counts.items() if v<5)})
 result={'run_dir':a.run_dir,'results':out,'recommended_window_size_for_next_experiment':10,'recommendation_reason':'Несколько окон одного scenario execution не являются полностью независимыми наблюдениями. Уменьшение размера окна не решает проблему недостатка независимых scenario executions.'};Path(a.json_report).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');Path(a.report).write_text('# Симуляция размеров окон\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
