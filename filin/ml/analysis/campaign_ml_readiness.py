from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'filin/ml/analysis'))
from campaign_provenance import audit_campaign_provenance
from campaign_execution_audit import audit_campaign_executions
from campaign_split_audit import audit_campaign_split
from campaign_common import write_result
def evaluate_readiness(output_root:Path)->dict:
 p=audit_campaign_provenance(output_root);e=audit_campaign_executions(output_root);s=audit_campaign_split(output_root);valid=p['campaign_provenance_valid'] and e['campaign_execution_valid'] and s['train_test_split_valid'];return {'window_pipeline_valid':valid,'campaign_provenance_valid':p['campaign_provenance_valid'],'campaign_execution_valid':e['campaign_execution_valid'],'train_test_split_valid':s['train_test_split_valid'],'minimum_train_support_valid':e['campaign_execution_valid'],'minimum_test_support_valid':e['campaign_execution_valid'],'temporal_scenarios_valid':not any('длительность' in x for x in e['errors']),'ready_for_ml':valid,'errors':p['errors']+e['errors']+s['errors']}
def main():
 p=argparse.ArgumentParser(description='Проверка готовности независимой кампании к ML.');p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--json-report');a=p.parse_args();r=evaluate_readiness(Path(a.output_root));write_result(Path(a.json_report) if a.json_report else None,r);print(json.dumps(r,ensure_ascii=False));raise SystemExit(0 if r['ready_for_ml'] else 1)
if __name__=='__main__':main()
