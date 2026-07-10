from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,recall_score,balanced_accuracy_score
from sklearn.dummy import DummyClassifier
def main():
 p=argparse.ArgumentParser(description='Независимая оценка client и Zeek sensor profiles.');p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);result={}
 for profile in ('client_core_v0_2','network_sensor_v0_3'):
  files=sorted(Path('filin/lab/output/datasets').glob(f'windows_{profile}_run_v030_zeek_*.csv'));frames=[pd.read_csv(x) for x in files];train=pd.concat([x for x in frames if '_train_' in x.run_id.iloc[0]]);test=pd.concat([x for x in frames if '_test_' in x.run_id.iloc[0]]);features=[x for x in train.columns if x not in {'run_id','label','label_type','scenario_id','execution_id','scenario_execution_key','feature_profile','observation_source','sensor_type','execution_mode','synthetic'} and pd.api.types.is_numeric_dtype(train[x])];build=lambda:Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]);folds=[]
  for run in sorted(train.run_id.unique()):
   model=build();fit=train[train.run_id!=run];valid=train[train.run_id==run];model.fit(fit[features],fit.label);prediction=model.predict(valid[features]);folds.append({'run_id':run,'macro_f1':f1_score(valid.label,prediction,average='macro',zero_division=0),'attack_macro_recall':recall_score(valid.label,prediction,labels=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'],average='macro',zero_division=0)})
  model=build();model.fit(train[features],train.label);pred=model.predict(test[features]);dummy=DummyClassifier(strategy='most_frequent').fit(train[features],train.label);dummy_pred=dummy.predict(test[features]);per={}
  for run in sorted(test.run_id.unique()):
   part=test[test.run_id==run];q=model.predict(part[features]);per[run]={'macro_f1':f1_score(part.label,q,average='macro',zero_division=0),'attack_macro_recall':recall_score(part.label,q,labels=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'],average='macro',zero_division=0)}
  result[profile]={'train_cv_folds':folds,'train_cv_macro_f1':sum(x['macro_f1'] for x in folds)/len(folds),'pooled_macro_f1':f1_score(test.label,pred,average='macro',zero_division=0),'dummy_macro_f1':f1_score(test.label,dummy_pred,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(test.label,pred),'attack_macro_recall':recall_score(test.label,pred,labels=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation'],average='macro',zero_division=0),'per_test_run':per,'rows_train':len(train),'rows_test':len(test)}
 (out/'sensor_baseline_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
