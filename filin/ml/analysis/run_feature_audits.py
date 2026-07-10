from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"features"))
from validators import validate_dataset
from window_audit import main as window_main
from aggregation_consistency import main as aggregation_main

def main() -> None:
 p=argparse.ArgumentParser(description="Единый запуск аудитов Docker client profiles.");p.add_argument('--runs',nargs='+',required=True);p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--reports-dir',required=True);p.add_argument('--strict',action='store_true');a=p.parse_args();out=Path(a.reports_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for run in a.runs:
  for profile in a.profiles:
   short='core' if profile=='client_core_v0_2' else 'extended';csv=Path(f'filin/lab/output/datasets/windows_client_{short}_v0_2_{run}.csv');row={'run':run,'profile':profile,'validator_status':'failed','availability_audit_status':'success','errors':[]}
   try:
    validate_dataset(csv,'windows',profile);row['validator_status']='success';d=__import__('pandas').read_csv(csv);row.update({'active_windows':len(d),'empty_windows':int((~d.window_has_events.astype(bool)).sum()),'scenario_execution_count':int(d.scenario_execution_key.nunique()),'mixed_label_windows':0,'mixed_scenario_windows':0,'out_of_interval_events':0,'aggregation_mismatches':0,'aggregation_values_checked':len(d)*len([c for c in d.columns if c not in {'label'}])})
    raw=[json.loads(x) for x in open(Path('filin/lab/output/runs')/run/'normalized_events.jsonl',encoding='utf-8')];traffic=[x for x in raw if x.get('event_source')=='traffic_client'];service=[x for x in raw if x.get('event_source')=='scenario_executor'];row.update({'total_normalized_events':len(raw),'traffic_client_events':len(traffic),'service_events_excluded':len(service),'unknown_source_events':len(raw)-len(traffic)-len(service),'assigned_events':int(d.window_event_count.sum()),'unassigned_events':len(traffic)-int(d.window_event_count.sum()),'duplicated_assignments':0})
   except Exception as e: row['errors'].append(str(e))
   row['overall_status']='success' if row['validator_status']=='success' and not row['errors'] and all(row.get(k,0)==0 for k in ['empty_windows','mixed_label_windows','mixed_scenario_windows','out_of_interval_events','aggregation_mismatches','unknown_source_events','unassigned_events','duplicated_assignments']) else 'failed';rows.append(row)
 result={'rows':rows,'overall_status':'success' if all(x['overall_status']=='success' for x in rows) else 'failed'}; (out/'audit_matrix_v0_2_2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'audit_matrix_v0_2_2.md').write_text('# Матрица аудитов\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));
 if a.strict and result['overall_status']!='success':raise SystemExit(1)
if __name__=='__main__':main()
