from __future__ import annotations
import argparse,json,os,subprocess,sys
from argparse import Namespace
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'filin/lab/tools'));sys.path.insert(0,str(ROOT/'filin/lab/campaigns'))
from run_lab_pipeline import run_pipeline
from scenario_runner import NATURAL_SCENARIO_ORDER
from campaign_schema import build_execution_metadata
def main():
 p=argparse.ArgumentParser(description='Запуск независимой кампании Zeek-наблюдений.');p.add_argument('--campaign',required=True);p.add_argument('--output-root',default='filin/lab/output');p.add_argument('--strict',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--run-ids',nargs='*');a=p.parse_args();c=yaml.safe_load(Path(a.campaign).read_text(encoding='utf-8'));out=Path(a.output_root);campaign_dir=c['campaign_id'].replace('-','_');sp=out/'campaigns'/campaign_dir/'status.json';status=json.loads(sp.read_text(encoding='utf-8')) if a.resume and sp.exists() else {}
 for i,run in enumerate(c['runs']):
  rid=run['run_id'];
  if a.run_ids and rid not in a.run_ids: continue
  if a.resume and status.get(rid)=='success': continue
  rd=out/'runs'/rid;sd=rd/'sensor';sd.mkdir(parents=True,exist_ok=True);env={**os.environ};subprocess.run(['docker','compose','-f','filin/lab/docker/docker-compose.lab.yml','up','-d','sensor-capture'],env=env,check=True)
  meta={'run_id':rid,'campaign_id':c['campaign_id'],'campaign_version':c['campaign_version'],'campaign_role':run['role'],'campaign_run_index':run['run_index'],'campaign_seed':run['random_seed']};variants={n:build_execution_metadata({'campaign_id':c['campaign_id'],'campaign_version':c['campaign_version']},run,n,s) for n,s in enumerate(NATURAL_SCENARIO_ORDER,1)}
  args=Namespace(run_dir=str(rd),scenarios='filin/lab/scenarios',base_time=f'2026-07-13T{10+i%8:02d}:00:00Z',schedule_mode='natural',gap_seconds=15,repeat=1,mock=False,docker=True,compose_file='filin/lab/docker/docker-compose.lab.yml',compose_project_dir='filin/lab/docker',time_scale=.05,random_seed=run['random_seed'],start_services=False,rebuild_services=False,stop_services_after_run=False,max_runtime_seconds=240,window_seconds=60)
  try:result=run_pipeline(args,meta,variants,True);subprocess.run(['docker','compose','-f','filin/lab/docker/docker-compose.lab.yml','stop','sensor-capture'],env=env,check=True);internal=f'captures/{rid}/attempt_001/capture.pcap';subprocess.run(['docker','run','--rm','-v','filin_sensor_capture:/captures','busybox','sh','-c',f'mkdir -p /captures/captures/{rid}/attempt_001 && cp /captures/capture.pcap /captures/{internal}'],check=True);subprocess.run([sys.executable,'filin/lab/sensor/run_zeek.py','--pcap',internal,'--output-dir',str(sd/'zeek'),'--storage-backend','docker_volume','--run-id',rid,'--strict'],check=True);subprocess.run([sys.executable,'filin/lab/sensor/normalize_zeek_events.py','--logs-dir',str(sd/'zeek'),'--output',str(sd/'zeek_events.jsonl'),'--run-id',rid],check=True);subprocess.run([sys.executable,'filin/lab/sensor/correlate_sensor_events.py','--manifest',str(rd/'scenario_manifest.yaml'),'--events',str(sd/'zeek_events.jsonl'),'--output',str(sd/'normalized_sensor_events.jsonl')],check=True);subprocess.run([sys.executable,'filin/ml/features/build_network_sensor_dataset.py','--manifest',str(rd/'scenario_manifest.yaml'),'--events',str(sd/'normalized_sensor_events.jsonl'),'--output',str(out/'datasets'/f'windows_network_sensor_v0_3_{rid}.csv')],check=True);status[rid]='success'
  except Exception as e:status[rid]=f'failed: {e}';
  sp.parent.mkdir(parents=True,exist_ok=True);sp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'campaigns'/campaign_dir).mkdir(parents=True,exist_ok=True);(out/'campaigns'/campaign_dir/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(status,ensure_ascii=False));
 if a.strict and any(v!='success' for v in status.values()):raise SystemExit(1)
if __name__=='__main__':main()
