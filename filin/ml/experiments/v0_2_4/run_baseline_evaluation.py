from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import joblib, numpy as np, yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'filin/ml/training'));sys.path.insert(0,str(ROOT/'filin/ml/analysis'))
from campaign_dataset_loader import load_campaign_datasets
from campaign_provenance import audit_campaign_provenance
from campaign_execution_audit import audit_campaign_executions
from campaign_split_audit import audit_campaign_split
from campaign_ml_readiness import evaluate_readiness

ATTACKS=['port_scan','auth_failures','web_probe','low_rate_dos','beacon_simulation']

def metrics(y, pred):
 labels=sorted(set(y)|set(pred)); p,r,f,s=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
 per={label:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(s[i])} for i,label in enumerate(labels)}
 attack=[per.get(label,{'recall':0,'f1':0}) for label in ATTACKS]
 collapsed_true=['attack' if v!='benign' else 'benign' for v in y];collapsed_pred=['attack' if v!='benign' else 'benign' for v in pred]
 cp,cr,cf,_=precision_recall_fscore_support(collapsed_true,collapsed_pred,labels=['attack'],average=None,zero_division=0)
 macro=precision_recall_fscore_support(y,pred,average='macro',zero_division=0); weighted=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
 return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_precision':float(macro[0]),'macro_recall':float(macro[1]),'macro_f1':float(macro[2]),'weighted_f1':float(weighted[2]),'attack_macro_recall':mean(v['recall'] for v in attack),'attack_macro_f1':mean(v['f1'] for v in attack),'collapsed_attack_precision':float(cp[0]),'collapsed_attack_recall':float(cr[0]),'collapsed_attack_f1':float(cf[0]),'per_class':per,'labels':labels,'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'zero_recall_classes':[k for k,v in per.items() if v['recall']==0]}

def candidates(random_state):
 return {'DummyClassifier':(DummyClassifier(strategy='most_frequent'),0),'LogisticRegression':(Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(C=1.0,class_weight='balanced',max_iter=1000,random_state=random_state))]),1),'RandomForestClassifier':(Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=1,max_features='sqrt',class_weight='balanced_subsample',random_state=random_state))]),2),'HistGradientBoostingClassifier':(Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(learning_rate=.1,max_iter=100,max_leaf_nodes=7,random_state=random_state))]),3)}

def choose(train,features,seed):
 rows=[]
 for name,(model,complexity) in candidates(seed).items():
  folds=[]
  for validation_run in sorted(train.run_id.unique()):
   fitting=train[train.run_id!=validation_run];validation=train[train.run_id==validation_run]
   model.fit(fitting[features],fitting.label); folds.append(metrics(validation.label,model.predict(validation[features])))
  rows.append({'name':name,'complexity':complexity,'folds':folds,'macro_f1':mean(x['macro_f1'] for x in folds),'attack_macro_recall':mean(x['attack_macro_recall'] for x in folds),'balanced_accuracy':mean(x['balanced_accuracy'] for x in folds),'macro_f1_std':pstdev(x['macro_f1'] for x in folds)})
 rows.sort(key=lambda x:(-x['macro_f1'],-x['attack_macro_recall'],-x['balanced_accuracy'],x['macro_f1_std'],x['complexity']))
 return rows, next(x for x in rows if x['name']!='DummyClassifier') if rows[0]['name']=='DummyClassifier' else rows[0]

def evaluate_profile(profile,index,output,artifacts,seed,policy):
 train,features,train_hashes=load_campaign_datasets(index,profile,'train');test,_,test_hashes=load_campaign_datasets(index,profile,'test');results,selected=choose(train,features,seed); model=candidates(seed)[selected['name']][0];model.fit(train[features],train.label); joblib.dump(model,artifacts/f'{profile}_technical_candidate.joblib')
 per={};predictions=[]
 for run in sorted(test.run_id.unique()):
  part=test[test.run_id==run];pred=model.predict(part[features]);per[run]=metrics(part.label,pred)
  predictions += [{'run_id':row.run_id,'execution_id':row.execution_id,'scenario_id':row.scenario_id,'scenario_variant_id':row.scenario_variant_id,'actual_label':row.label,'predicted_label':str(value),'window_event_count':row.window_event_count,'window_duration_seconds':row.window_duration_seconds} for row,value in zip(part.itertuples(),pred)]
 pooled_pred=model.predict(test[features]);pooled=metrics(test.label,pooled_pred);dummy=DummyClassifier(strategy='most_frequent').fit(train[features],train.label);dummy_metrics=metrics(test.label,dummy.predict(test[features]))
 gains={key:pooled[key]-dummy_metrics[key] for key in ('macro_f1','balanced_accuracy','attack_macro_recall','collapsed_attack_recall')};cfg=policy['useful_model_policy'];useful=gains['macro_f1']>=cfg['minimum_macro_f1_gain_over_dummy'] and gains['balanced_accuracy']>=cfg['minimum_balanced_accuracy_gain_over_dummy'] and pooled['attack_macro_recall']>=cfg['minimum_attack_macro_recall'] and sum(v['recall']>0 for k,v in pooled['per_class'].items() if k!='benign')>=cfg['minimum_attack_classes_with_nonzero_recall'] and pooled['collapsed_attack_recall']>=cfg['minimum_collapsed_attack_recall']
 value={'profile':profile,'selected_model':selected['name'],'train_cv':results,'pooled_test':pooled,'dummy_test':dummy_metrics,'gains':gains,'useful_model_found':useful,'per_test_run':per,'stable_across_test_runs':all(x['attack_macro_recall']>0 for x in per.values()) and pstdev(x['macro_f1'] for x in per.values())<=.15,'train_sha256':train_hashes,'test_sha256':test_hashes,'features':features,'predictions':predictions};(output/f'{profile}_evaluation.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');return value

def main():
 p=argparse.ArgumentParser(description='Оценка baseline-моделей независимой кампании.');p.add_argument('--config',required=True);p.add_argument('--policy',required=True);p.add_argument('--campaign-index',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--strict',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'));policy=yaml.safe_load(Path(a.policy).read_text(encoding='utf-8'));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);art=ROOT/'filin/ml/artifacts/v0_2_4';art.mkdir(parents=True,exist_ok=True);root=ROOT/'filin/lab/output';integrity={'provenance':audit_campaign_provenance(root),'execution':audit_campaign_executions(root),'split':audit_campaign_split(root),'readiness':evaluate_readiness(root)}
 if not all(not item.get('errors') for item in integrity.values()):raise ValueError('Integrity audit кампании не пройден.')
 campaign_dir=Path(a.campaign_index).parent; dataset_index={**json.loads((campaign_dir/'train_datasets.json').read_text(encoding='utf-8')) , **json.loads((campaign_dir/'test_datasets.json').read_text(encoding='utf-8'))}; combined_index=out/'dataset_index_runtime.json'; combined_index.write_text(json.dumps(dataset_index,ensure_ascii=False),encoding='utf-8')
 profiles=[evaluate_profile(profile,combined_index,out,art,config['random_state'],policy) for profile in config['feature_profiles']]
 summary={'evaluation_pipeline_valid':True,'external_test_completed':True,'integrity':integrity,'profiles':profiles,'useful_model_found':any(x['useful_model_found'] for x in profiles),'recommended_feature_profile':max(profiles,key=lambda x:x['pooled_test']['macro_f1'])['profile']};(out/'experiment_metadata.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(out/'baseline_evaluation_summary.md').write_text('# Филин v0.2.4 — baseline evaluation\n\n'+json.dumps({k:v for k,v in summary.items() if k!='profiles'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,default=str))
if __name__=='__main__':main()
