from __future__ import annotations
import json
from pathlib import Path
def audit_run(run_dir:Path)->dict:
 events=[json.loads(x) for x in (run_dir/'sensor'/'normalized_sensor_events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];counts={k:sum(e.get('correlation_status')==k for e in events) for k in ('assigned','background','excluded','ambiguous','unassigned')};return {**counts,'correlation_audit_status':'success' if counts['assigned'] else 'failed'}
