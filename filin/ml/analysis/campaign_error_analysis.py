from __future__ import annotations
from collections import Counter
def analyze_predictions(rows):
    errors=[row for row in rows if row['actual_label']!=row['predicted_label']]
    return {'error_count':len(errors),'attacks_as_benign':sum(row['actual_label']!='benign' and row['predicted_label']=='benign' for row in errors),'benign_as_attack':sum(row['actual_label']=='benign' and row['predicted_label']!='benign' for row in errors),'by_pair':Counter(f"{row['actual_label']}->{row['predicted_label']}" for row in errors)}
