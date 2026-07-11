from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,balanced_accuracy_score,recall_score
def main():
 p=argparse.ArgumentParser(description='Внешняя оценка frozen Zeek baseline.');p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);root=Path('filin/lab/output/datasets');train=pd.concat([pd.read_csv(x) for x in root.glob('windows_network_sensor_v0_3_run_v030_zeek_train_*.csv')]);robust=pd.concat([pd.read_csv(x) for x in root.glob('windows_network_sensor_v0_3_run_v032_*.csv')]);excluded={'run_id','label','label_type','scenario_id','execution_id','scenario_execution_key','feature_profile','observation_source','sensor_type','execution_mode','synthetic'};features=[x for x in train if x not in excluded and pd.api.types.is_numeric_dtype(train[x])];model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]);model.fit(train[features],train.label);per={}
 for run in sorted(robust.run_id.unique()):
  frame=robust[robust.run_id==run];pred=model.predict(frame[features]);per[run]={'macro_f1':f1_score(frame.label,pred,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(frame.label,pred),'attack_macro_recall':recall_score(frame.label,pred,labels=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'],average='macro',zero_division=0)}
 pred=model.predict(robust[features]);result={'frozen_model_class':'LogisticRegression','robustness_rows':len(robust),'per_run':per,'pooled_macro_f1':f1_score(robust.label,pred,average='macro',zero_division=0),'pooled_balanced_accuracy':balanced_accuracy_score(robust.label,pred),'pooled_attack_macro_recall':recall_score(robust.label,pred,labels=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'],average='macro',zero_division=0),'model_retrained_on_robustness_data':False};(out/'robustness_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
