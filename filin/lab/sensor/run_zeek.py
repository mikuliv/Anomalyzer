from __future__ import annotations
import argparse,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description='Offline-обработка PCAP средством Zeek.');p.add_argument('--pcap',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--strict',action='store_true');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);r=subprocess.run(['docker','run','--rm','-v',f'{Path(a.pcap).resolve()}:/input/capture.pcap:ro','-v',f'{out.resolve()}:/output','zeek/zeek:latest','sh','-c','cd /output && zeek -C -r /input/capture.pcap LogAscii::use_json=T'],capture_output=True,text=True);print(r.stdout);print(r.stderr)
 if a.strict and (r.returncode or not (out/'conn.log').exists()):raise SystemExit(1)
if __name__=='__main__':main()
