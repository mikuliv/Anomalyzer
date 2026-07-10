from __future__ import annotations
import subprocess
from dataclasses import dataclass
@dataclass
class SensorArtifactStorage:
    volume_name:str='filin_sensor_capture'
    def _run(self,args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip()
    def pcap_exists(self,path):return self._run(['docker','run','--rm','-v',f'{self.volume_name}:/captures:ro','busybox','sh','-c',f'test -s /captures/{path}'])==''
    def pcap_size(self,path):return int(self._run(['docker','run','--rm','-v',f'{self.volume_name}:/captures:ro','busybox','sh','-c',f'wc -c < /captures/{path}']))
    def pcap_sha256(self,path):return self._run(['docker','run','--rm','-v',f'{self.volume_name}:/captures:ro','busybox','sha256sum',f'/captures/{path}']).split()[0]
    def run_zeek(self,pcap_path,run_id,attempt_id='attempt_001'):
        output_volume='filin_sensor_zeek'; command=['docker','run','--rm','-v',f'{self.volume_name}:/captures:ro','-v',f'{output_volume}:/zeek-output','zeek/zeek:latest','sh','-c',f'mkdir -p /zeek-output/{run_id}/{attempt_id} && cd /zeek-output/{run_id}/{attempt_id} && zeek -C -r /captures/{pcap_path} LogAscii::use_json=T']
        return subprocess.run(command,check=False,capture_output=True,text=True)
