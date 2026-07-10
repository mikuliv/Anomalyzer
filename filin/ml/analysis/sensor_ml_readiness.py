from __future__ import annotations
import json
from pathlib import Path
def evaluate(status:dict)->dict:return {'ready_for_sensor_ml':len(status)==9 and all(v=='success' for v in status.values()),'run_count':len(status),'ml_training_started':False}
