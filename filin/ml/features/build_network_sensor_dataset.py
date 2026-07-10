from __future__ import annotations
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
import yaml
from schema import NETWORK_SENSOR_V0_3
def num(v):
 try:return float(v or 0)
 except:return 0.0
def aggregate(events):
 conn=[e for e in events if e.get('sensor_log_type')=='conn'];http=[e for e in events if e.get('sensor_log_type')=='http'];dns=[e for e in events if e.get('sensor_log_type')=='dns'];raw=[e.get('raw',{}) for e in conn];ob=sum(num(x.get('orig_bytes')) for x in raw);rb=sum(num(x.get('resp_bytes')) for x in raw);op=sum(num(x.get('orig_pkts')) for x in raw);rp=sum(num(x.get('resp_pkts')) for x in raw);dur=[num(x.get('duration')) for x in raw];states=[x.get('conn_state') for x in raw];success=sum(x in {'SF','S1','S2','S3'} for x in states);failed=len(conn)-success
 return {k:0.0 for k in NETWORK_SENSOR_V0_3}|{'flow_count':float(len(conn)),'tcp_flow_count':float(sum(x.get('proto')=='tcp' for x in raw)),'udp_flow_count':float(sum(x.get('proto')=='udp' for x in raw)),'unique_destination_ip_count':float(len({x.get('id.resp_h') for x in raw})),'unique_destination_port_count':float(len({x.get('id.resp_p') for x in raw})),'flow_duration_mean':sum(dur)/len(dur) if dur else 0,'flow_duration_median':sorted(dur)[len(dur)//2] if dur else 0,'flow_duration_max':max(dur,default=0),'orig_bytes_total':ob,'resp_bytes_total':rb,'total_bytes':ob+rb,'orig_bytes_mean':ob/len(conn) if conn else 0,'resp_bytes_mean':rb/len(conn) if conn else 0,'orig_resp_bytes_ratio':ob/rb if rb else 0,'orig_packets_total':op,'resp_packets_total':rp,'total_packets':op+rp,'orig_packets_mean':op/len(conn) if conn else 0,'resp_packets_mean':rp/len(conn) if conn else 0,'orig_resp_packets_ratio':op/rp if rp else 0,'successful_connection_count':float(success),'failed_connection_count':float(failed),'connection_success_rate':success/len(conn) if conn else 0,'connection_failure_rate':failed/len(conn) if conn else 0,'unique_conn_state_count':float(len(set(states))),'http_request_count':float(len(http)),'http_get_count':float(sum(e.get('raw',{}).get('method')=='GET' for e in http)),'http_post_count':float(sum(e.get('raw',{}).get('method')=='POST' for e in http)),'dns_query_count':float(len(dns))}
def main():
 p=argparse.ArgumentParser(description='Построение network sensor dataset из correlated Zeek events.');p.add_argument('--manifest',required=True);p.add_argument('--events',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=yaml.safe_load(Path(a.manifest).read_text(encoding='utf-8'));ev=[json.loads(x) for x in Path(a.events).read_text(encoding='utf-8').splitlines() if x.strip()];rows=[]
 for s in m['scenarios']:
  selected=[e for e in ev if e.get('execution_id')==s.get('execution_id') and e.get('correlation_status')=='assigned'];row={'run_id':m['run_id'],'execution_id':s.get('execution_id'),'scenario_execution_key':f"{m['run_id']}:{s['run_sequence']}:{s['scenario_id']}",'window_index':0,'scenario_id':s['scenario_id'],'label':s['label'],'label_type':s['type'],'execution_mode':'docker','synthetic':False,'observation_source':'network_sensor','sensor_type':'zeek','feature_profile':'network_sensor_v0_3','window_event_count':len(selected),'window_has_events':bool(selected),'window_duration_seconds':1};row.update(aggregate(selected));rows.append(row)
 fields=list(rows[0]);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with Path(a.output).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
