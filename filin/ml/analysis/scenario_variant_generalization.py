from __future__ import annotations
import pandas as pd
def compare_variants(train:pd.DataFrame,test:pd.DataFrame)->dict:
    return {'parameter_hash_overlap':sorted(set(train.scenario_parameter_hash)&set(test.scenario_parameter_hash)),'exact_feature_duplicates':int(test.merge(train,on=[c for c in train.columns if c.startswith(('event_','http_','auth_','tcp_','dns_','bytes_','latency_'))],how='inner').shape[0])}
