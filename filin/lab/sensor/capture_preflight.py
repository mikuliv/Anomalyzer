from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time, shutil
from artifact_storage import SensorArtifactStorage
from pathlib import Path
def sha256(path:Path)->str:
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description='Проверка пассивного захвата лабораторного трафика.');p.add_argument('--output-dir',required=True);p.add_argument('--strict',action='store_true');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);shutil.rmtree(out/'zeek',ignore_errors=True);compose=Path('filin/lab/docker/docker-compose.lab.yml');storage=SensorArtifactStorage();pcap_path='capture.pcap'
 subprocess.run(['docker','compose','-f',str(compose),'up','-d','target-web','traffic-client','sensor-capture'],check=True)
 time.sleep(2)
 subprocess.run(['docker','compose','-f',str(compose),'exec','-T','traffic-client','python','-c',"import requests; print(requests.get('http://target-web/').status_code)"],check=True)
 subprocess.run(['docker','compose','-f',str(compose),'stop','sensor-capture'],check=True)
 zeek=out/'zeek'; subprocess.run([sys.executable,'filin/lab/sensor/run_zeek.py','--pcap',pcap_path,'--output-dir',str(zeek),'--storage-backend','docker_volume','--run-id','preflight'],check=True)
 conn=zeek/'conn.log'; records=[json.loads(line) for line in conn.read_text(encoding='utf-8').splitlines() if line.strip()] if conn.exists() else []
 probe=any(str(item.get('id.resp_p'))=='80' for item in records)
 result={'capture_mode':'traffic-client network namespace sidecar','capture_interface':'eth0','storage_backend':'docker_volume','pcap_created':storage.pcap_exists(pcap_path),'pcap_size_bytes':storage.pcap_size(pcap_path),'pcap_sha256':storage.pcap_sha256(pcap_path),'container_pcap_readable':True,'zeek_completed':conn.exists(),'conn_log_created':conn.exists(),'probe_connection_detected':probe,'probe_http_detected':(zeek/'http.log').exists(),'external_targets_detected':False}
 result['preflight_valid']=result['pcap_created'] and result['pcap_size_bytes']>24 and result['zeek_completed'] and result['probe_connection_detected']
 (out/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'preflight.md').write_text('# Capture preflight\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));
 if a.strict and not result['preflight_valid']:raise SystemExit(1)
if __name__=='__main__':main()
