from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'filin/ml/analysis'))
from campaign_common import sha256, write_result
def audit_campaign_split(output_root: Path)->dict:
 status=json.loads((output_root/'campaigns/filin_v0_2_3/campaign_status.json').read_text(encoding='utf-8'));train={k:v for k,v in status['runs'].items() if v['role']=='train'};test={k:v for k,v in status['runs'].items() if v['role']=='test'};errors=[]
 if set(train)&set(test):errors.append('Пересекаются run IDs')
 th={h for v in train.values() for h in v.get('artifacts',{}).values()}; eh={h for v in test.values() for h in v.get('artifacts',{}).values()}
 if th&eh:errors.append('Пересекаются hashes artifacts')
 return {'train_test_split_valid':not errors,'train_runs':sorted(train),'test_runs':sorted(test),'errors':errors}
def main():
 p=argparse.ArgumentParser(description='Аудит разделения train/test кампании.');p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--json-report');a=p.parse_args();r=audit_campaign_split(Path(a.output_root));write_result(Path(a.json_report) if a.json_report else None,r);print(json.dumps(r,ensure_ascii=False));raise SystemExit(0 if r['train_test_split_valid'] else 1)
if __name__=='__main__':main()
