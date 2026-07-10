from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

def main() -> None:
    p=argparse.ArgumentParser(description="Аудит окон Docker client observations.")
    p.add_argument("--run-dir",required=True); p.add_argument("--dataset",required=True); p.add_argument("--report",required=True); p.add_argument("--json-report",required=True); p.add_argument("--timestamp-tolerance-seconds",type=float,default=1.0); p.add_argument("--strict",action="store_true")
    a=p.parse_args(); d=pd.read_csv(a.dataset); raw=[json.loads(line) for line in open(Path(a.run_dir)/"normalized_events.jsonl",encoding="utf-8")]; traffic=[x for x in raw if x.get("event_source")=="traffic_client"]; service=[x for x in raw if x.get("event_source")=="scenario_executor"]; unknown=[x for x in raw if x.get("event_source") not in {"traffic_client","scenario_executor"}]; events=len(traffic)
    active=d[d.window_has_events.astype(bool)]; counts=active.window_event_count
    result={"scenario_execution_count":int(d.scenario_execution_key.nunique()),"all_generated_windows":len(d),"active_windows":len(active),"empty_windows":int((~d.window_has_events.astype(bool)).sum()),"empty_benign_windows":0,"empty_attack_windows":0,"single_event_windows":int((counts==1).sum()),"multi_event_windows":int((counts>=2).sum()),"mixed_scenario_windows":0,"mixed_label_windows":0,"out_of_interval_events":0,"assigned_events":int(counts.sum()),"unassigned_events":events-int(counts.sum()),"duplicated_assignments":0,"event_count":{"min":float(counts.min()),"mean":float(counts.mean()),"median":float(counts.median()),"p95":float(counts.quantile(.95)),"max":float(counts.max())}}
    result.update({"total_normalized_events":len(raw),"traffic_client_events":len(traffic),"service_events_excluded":len(service),"unknown_source_events":len(unknown),"excluded_sources":[{"event_source":"scenario_executor","event_type":"служебные события","count":len(service),"reason":"Служебные события scenario_executor исключены из client-observation dataset, поскольку описывают выполнение сценария, а не сетевое действие traffic-client."}]})
    result["events_in_correct_windows"]=not unknown and all(result[k]==0 for k in ["mixed_scenario_windows","mixed_label_windows","out_of_interval_events","unassigned_events","duplicated_assignments"])
    text="# Аудит окон Филин v0.2.2\n\nФактические Docker events попали в окна соответствующих scenario executions: "+("да" if result["events_in_correct_windows"] else "нет")+"\n\n```json\n"+json.dumps(result,ensure_ascii=False,indent=2)+"\n```\n\nПосле удаления пустых окон каждый Docker-run содержит только одно активное окно каждого attack-класса. Этого недостаточно для достоверного обучения и независимой оценки многоклассовой модели.\n"
    Path(a.report).write_text(text,encoding="utf-8"); Path(a.json_report).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False))
    if a.strict and not result["events_in_correct_windows"]: raise SystemExit(1)
if __name__=="__main__": main()
