from __future__ import annotations
import json
from pathlib import Path
def parse_zeek_log(path:Path):
 text=path.read_text(encoding='utf-8',errors='replace').splitlines()
 if text and text[0].lstrip().startswith('{'):return [json.loads(line) for line in text if line.strip()]
 fields=[];records=[]
 for line in text:
  if line.startswith('#fields'):fields=line.split('\t')[1:]
  elif line and not line.startswith('#') and fields:records.append(dict(zip(fields,line.split('\t'))))
 return records
