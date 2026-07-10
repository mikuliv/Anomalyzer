from __future__ import annotations
from pathlib import Path
def audit_run(run_dir:Path)->dict:
 p=run_dir/'sensor';return {'capture_audit_status':'success' if (p/'zeek'/'conn.log').exists() else 'failed'}
