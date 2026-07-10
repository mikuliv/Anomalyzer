from __future__ import annotations
def summarize_importance(values:dict[str,float],top_n:int=20)->list[tuple[str,float]]:
    return sorted(values.items(),key=lambda item:abs(item[1]),reverse=True)[:top_n]
