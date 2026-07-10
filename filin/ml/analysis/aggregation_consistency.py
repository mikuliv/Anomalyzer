from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"features"))
from build_windows_dataset import aggregate_client_window,parse_time
from schema import get_feature_profile
def main() -> None:
 p=argparse.ArgumentParser(description="Проверка согласованности агрегатов client observations.");p.add_argument("--run-dir",required=True);p.add_argument("--dataset",required=True);p.add_argument("--samples-per-class",type=int,default=3);p.add_argument("--all-windows",action="store_true");p.add_argument("--report",required=True);p.add_argument("--json-report",required=True);p.add_argument("--strict",action="store_true");a=p.parse_args();d=pd.read_csv(a.dataset); profile=d.feature_profile.iloc[0]; events=[json.loads(x) for x in open(Path(a.run_dir)/"normalized_events.jsonl",encoding="utf-8") if json.loads(x).get("event_source")=="traffic_client"]; rows=d if a.all_windows else d.groupby("label",group_keys=False).head(a.samples_per_class); mism=[]; checked=0
 for _,row in rows.iterrows():
  start,end=parse_time(row.window_start),parse_time(row.window_end); es=[e for e in events if e.get("run_sequence")==row.run_sequence and e.get("scenario_id")==row.scenario_id and start<=parse_time(e["timestamp"])<end]; actual=aggregate_client_window(es,float(row.window_duration_seconds))
  for f in get_feature_profile(profile):
   checked+=1; x,y=row[f],actual[f]
   if not ((pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and abs(float(x)-float(y))<=1e-6)): mism.append({"feature":f,"expected":x,"actual":y,"window":row.scenario_execution_key})
 r={"checked_windows":len(rows),"checked_features":checked,"aggregation_mismatches":len(mism),"mismatches":mism};Path(a.json_report).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");Path(a.report).write_text("# Проверка агрегации\n\n```json\n"+json.dumps(r,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8");print(json.dumps(r,ensure_ascii=False));
 if a.strict and mism:raise SystemExit(1)
if __name__=="__main__":main()
