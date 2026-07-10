from __future__ import annotations
import argparse,json
from pathlib import Path
def build_index(output_root:Path)->dict:
 status=json.loads((output_root/'campaigns/filin_v0_2_3/campaign_status.json').read_text(encoding='utf-8'));result={}
 for profile in ('client_core_v0_2','client_extended_v0_2'):
  for role in ('train','test'):
   runs=[run for run,item in status['runs'].items() if item['role']==role and item['status']=='success'];result[f'{profile}_{role}']=[str(output_root/'datasets'/f'windows_{profile}_{run}.csv') for run in runs]
 campaign_dir=output_root/'campaigns/filin_v0_2_3';campaign_dir.mkdir(parents=True,exist_ok=True)
 for name,paths in result.items():(campaign_dir/f'{name}_datasets.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
 (campaign_dir/'train_datasets.json').write_text(json.dumps({k:v for k,v in result.items() if k.endswith('_train')},ensure_ascii=False,indent=2),encoding='utf-8');(campaign_dir/'test_datasets.json').write_text(json.dumps({k:v for k,v in result.items() if k.endswith('_test')},ensure_ascii=False,indent=2),encoding='utf-8')
 return result
def main():
 p=argparse.ArgumentParser(description='Построение индекса datasets кампании.');p.add_argument('--output-root',default='filin/lab/output');a=p.parse_args();r=build_index(Path(a.output_root));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
