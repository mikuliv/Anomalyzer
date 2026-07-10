from __future__ import annotations

from typing import Any


def drift_level(psi: float, constant_changed: bool) -> str:
    if constant_changed or psi >= 0.25:
        return "сильное"
    if psi >= 0.10:
        return "заметное"
    return "слабое"


def write_drift_report(result: dict[str, Any]) -> str:
    lines = ["# Анализ смещения признаков Филин v0.2.1", "", "## Источники", "", f"- Reference: `{result['reference']}`", f"- Comparison: `{result['comparison']}`", f"- Reference source: {result['reference_source']}", f"- Comparison source: {result['comparison_source']}", "", "Границы PSI являются ориентировочными и не доказывают ухудшение модели сами по себе.", "", "## Признаки с наибольшим смещением", "", "| Признак | PSI | Standardized mean difference | Zero-rate difference | Уровень |", "| --- | ---: | ---: | ---: | --- |"]
    for item in result["features"][: result["top_n"]]:
        lines.append(f"| {item['feature']} | {item['population_stability_index']:.4f} | {item['standardized_mean_difference']:.4f} | {item['zero_rate_difference']:.4f} | {item['drift_level']} |")
    if result.get("class_warnings"):
        lines.extend(["", "## Предупреждения по классам", "", *[f"- {warning}" for warning in result["class_warnings"]]])
    return "\n".join(lines) + "\n"
