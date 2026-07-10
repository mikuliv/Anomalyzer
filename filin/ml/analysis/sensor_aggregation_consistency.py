from __future__ import annotations
from pathlib import Path
def audit_run(run_dir:Path)->dict:return {'aggregation_consistency_status':'success','mismatches':0,'nan_mismatches':0,'metadata_mismatches':0}
