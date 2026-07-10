from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'filin/ml/analysis'))
from campaign_ml_readiness import evaluate_readiness
def build_summary(output_root:Path,output:Path)->None:
 r=evaluate_readiness(output_root);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('# Филин v0.2.3 — независимые laboratory executions\n\n'+f"- window_pipeline_valid: {str(r['window_pipeline_valid']).lower()}\n- ready_for_ml: {str(r['ready_for_ml']).lower()}\n\nКампания использует разные фактические Docker-runs для train и test. Увеличение количества окон одного scenario execution не увеличивает количество независимых наблюдений.\n",encoding='utf-8')
def main():
 p=argparse.ArgumentParser(description='Сводный отчёт кампании.');p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--output',default='filin/lab/output/campaigns/filin_v0_2_3/campaign_summary.md');a=p.parse_args();build_summary(Path(a.output_root),Path(a.output))
if __name__=='__main__':main()
