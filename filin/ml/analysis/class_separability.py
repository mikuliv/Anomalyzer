from __future__ import annotations
import pandas as pd
def describe_class_support(frame:pd.DataFrame)->dict:
    return {'support':{str(k):int(v) for k,v in frame['label'].value_counts().items()}}
