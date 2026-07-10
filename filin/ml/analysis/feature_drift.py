from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))
from schema import get_model_feature_columns  # noqa: E402
from report_writer import drift_level, write_drift_report  # noqa: E402


def source_name(df: pd.DataFrame) -> str:
    mode = set(df.get("execution_mode", pd.Series(dtype=str)).astype(str))
    observation = set(df.get("observation_source", pd.Series(dtype=str)).astype(str))
    if mode == {"docker"} and observation == {"client"}:
        return "события, полученные при реальном выполнении действий между контейнерами и наблюдаемые со стороны traffic-client"
    return "mock generator" if mode == {"mock"} else "лабораторный источник неизвестного типа"


def statistics(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    count = len(numeric)
    if valid.empty:
        return {"count": count, "missing_count": count, "missing_rate": 1.0, "zero_count": 0, "zero_rate": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "p75": 0.0, "max": 0.0}
    return {"count": count, "missing_count": int(numeric.isna().sum()), "missing_rate": float(numeric.isna().mean()), "zero_count": int((valid == 0).sum()), "zero_rate": float((valid == 0).mean()), "mean": float(valid.mean()), "median": float(valid.median()), "std": float(valid.std(ddof=0)), "min": float(valid.min()), "p25": float(valid.quantile(.25)), "p75": float(valid.quantile(.75)), "max": float(valid.max())}


def psi(reference: pd.Series, comparison: pd.Series, epsilon: float) -> tuple[float, bool]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    comp = pd.to_numeric(comparison, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(comp) == 0:
        return 0.0, False
    if np.all(ref == ref[0]):
        return (0.0, False) if np.all(comp == ref[0]) else (1.0, True)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0, False
    edges[0], edges[-1] = -np.inf, np.inf
    ref_bins = np.histogram(ref, bins=edges)[0] / len(ref)
    comp_bins = np.histogram(comp, bins=edges)[0] / len(comp)
    ref_bins, comp_bins = np.clip(ref_bins, epsilon, None), np.clip(comp_bins, epsilon, None)
    return float(np.sum((comp_bins - ref_bins) * np.log(comp_bins / ref_bins))), False


def compare_feature(reference: pd.Series, comparison: pd.Series, epsilon: float) -> dict[str, Any]:
    ref, comp = statistics(reference), statistics(comparison)
    pooled = float(np.sqrt((float(ref["std"]) ** 2 + float(comp["std"]) ** 2) / 2))
    psi_value, constant_changed = psi(reference, comparison, epsilon)
    return {"reference": ref, "comparison": comp, "absolute_mean_difference": abs(float(comp["mean"]) - float(ref["mean"])), "relative_mean_difference": abs(float(comp["mean"]) - float(ref["mean"])) / max(abs(float(ref["mean"])), epsilon), "absolute_median_difference": abs(float(comp["median"]) - float(ref["median"])), "relative_median_difference": abs(float(comp["median"]) - float(ref["median"])) / max(abs(float(ref["median"])), epsilon), "standardized_mean_difference": 0.0 if pooled == 0 else (float(comp["mean"]) - float(ref["mean"])) / pooled, "zero_rate_difference": float(comp["zero_rate"]) - float(ref["zero_rate"]), "missing_rate_difference": float(comp["missing_rate"]) - float(ref["missing_rate"]), "population_stability_index": psi_value, "constant_changed": constant_changed, "drift_level": drift_level(psi_value, constant_changed)}


def analyze(reference: pd.DataFrame, comparison: pd.DataFrame, target: str, top_n: int, epsilon: float, by_class: bool) -> dict[str, Any]:
    if target not in reference.columns or target not in comparison.columns:
        raise ValueError(f"Целевая колонка не найдена: {target}")
    features = [column for column in get_model_feature_columns(reference.columns) if column in comparison.columns and pd.api.types.is_numeric_dtype(reference[column])]
    if not features:
        raise ValueError("Не найдены общие числовые model features.")
    rows = []
    for feature in features:
        row = {"feature": feature, **compare_feature(reference[feature], comparison[feature], epsilon)}
        rows.append(row)
    rows.sort(key=lambda item: (item["population_stability_index"], abs(item["standardized_mean_difference"])), reverse=True)
    result: dict[str, Any] = {"reference_source": source_name(reference), "comparison_source": source_name(comparison), "features": rows, "top_n": top_n, "class_warnings": []}
    if by_class:
        result["by_class"] = {}
        ref_labels, comp_labels = set(reference[target].astype(str)), set(comparison[target].astype(str))
        for label in sorted(ref_labels | comp_labels):
            if label not in ref_labels or label not in comp_labels:
                result["class_warnings"].append(f"Класс {label} присутствует только в одном наборе.")
                continue
            result["by_class"][label] = [{"feature": feature, **compare_feature(reference.loc[reference[target].astype(str) == label, feature], comparison.loc[comparison[target].astype(str) == label, feature], epsilon)} for feature in features]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ смещения числовых признаков между laboratory datasets.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--target", default="label")
    parser.add_argument("--report", required=True)
    parser.add_argument("--json-report", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--by-class", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-9)
    args = parser.parse_args()
    reference_path, comparison_path = Path(args.reference), Path(args.comparison)
    result = analyze(pd.read_csv(reference_path), pd.read_csv(comparison_path), args.target, args.top_n, args.epsilon, args.by_class)
    result.update({"reference": str(reference_path), "comparison": str(comparison_path)})
    Path(args.report).write_text(write_drift_report(result), encoding="utf-8")
    Path(args.json_report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Drift-отчёт: {args.report}")


if __name__ == "__main__":
    main()
