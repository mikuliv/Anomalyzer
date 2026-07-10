from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description='Offline-обработка PCAP средством Zeek.');p.add_argument('--pcap',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--strict',action='store_true');p.add_argument('--storage-backend',default='host_filesystem',choices=('host_filesystem','docker_volume'));p.add_argument('--run-id');p.add_argument('--attempt-id',default='attempt_001');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 if a.storage_backend=='docker_volume':
  sys.path.insert(0,str(Path(__file__).parent));from artifact_storage import SensorArtifactStorage;r=SensorArtifactStorage().run_zeek(a.pcap,a.run_id or 'run',a.attempt_id)
  if r.returncode==0:
   subprocess.run(['docker','run','--rm','-v','filin_sensor_zeek:/zeek:ro','-v',f'{out.resolve()}:/export','busybox','sh','-c',f'cp -a /zeek/{a.run_id or "run"}/{a.attempt_id}/. /export/'],check=True,capture_output=True,text=True)
 else:r=subprocess.run(['docker','run','--rm','-v',f'{Path(a.pcap).resolve()}:/input/capture.pcap:ro','-v',f'{out.resolve()}:/output','zeek/zeek:latest','sh','-c','cd /output && zeek -C -r /input/capture.pcap LogAscii::use_json=T'],capture_output=True,text=True)
 if a.strict and (r.returncode or not (out/'conn.log').exists()):raise SystemExit(1)
if __name__=='__main__':main()
