from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"features"))
from schema import PACKET_FEATURES,get_feature_profile
def main() -> None:
 p=argparse.ArgumentParser(description="Аудит доступности client-признаков."); p.add_argument("--dataset",required=True);p.add_argument("--feature-profile",required=True);p.add_argument("--target",default="label");p.add_argument("--report",required=True);p.add_argument("--json-report",required=True);a=p.parse_args();d=pd.read_csv(a.dataset); profile=get_feature_profile(a.feature_profile); rows=[]
 for f in profile:
  s=pd.to_numeric(d[f],errors="coerce"); rows.append({"feature":f,"missing_rate":float(s.isna().mean()),"zero_rate":float((s.dropna()==0).mean()) if len(s.dropna()) else 0,"constant":bool(s.nunique(dropna=False)<=1),"all_zero":bool((s.dropna()==0).all()) if len(s.dropna()) else False})
 r={"all_zero_features":[x["feature"] for x in rows if x["all_zero"]],"constant_features":[x["feature"] for x in rows if x["constant"]],"almost_all_zero_features":[x["feature"] for x in rows if x["zero_rate"]>=.95],"packet_features_in_profile":sorted(set(profile)&PACKET_FEATURES),"features":rows}
 if r["packet_features_in_profile"]: raise ValueError("Packet-признаки попали в client profile")
 Path(a.json_report).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");Path(a.report).write_text("# Доступность признаков\n\n```json\n"+json.dumps(r,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
