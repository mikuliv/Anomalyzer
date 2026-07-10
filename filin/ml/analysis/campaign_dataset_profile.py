from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def profile_campaign_datasets(output_root:Path)->dict:
 result={}
 for file in (output_root/'datasets').glob('windows_client_*_run_v023_*.csv'):
  with file.open(encoding='utf-8',newline='') as source: result[file.name]={'rows':sum(1 for _ in csv.DictReader(source))}
 return {'dataset_profiles':result}
def main():
 p=argparse.ArgumentParser(description='Профиль datasets кампании.');p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--json-report');a=p.parse_args();r=profile_campaign_datasets(Path(a.output_root));
 if a.json_report:Path(a.json_report).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
