from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_info(run_name: str) -> dict[str, Any]:
    run_dir = REPO_ROOT / "filin" / "lab" / "output" / "runs" / run_name
    manifest = yaml.safe_load((run_dir / "scenario_manifest.yaml").read_text(encoding="utf-8")) or {}
    dataset = pd.read_csv(REPO_ROOT / "filin" / "lab" / "output" / "datasets" / f"windows_v0_1_{run_name}.csv")
    flows = pd.read_csv(REPO_ROOT / "filin" / "lab" / "output" / "datasets" / f"flows_v0_1_{run_name}.csv")
    return {"name": run_name, "manifest": manifest, "windows_rows": len(dataset), "flows_rows": len(flows), "classes": dict(Counter(dataset["label"]))}


def experiment_metadata(train: str, test: str) -> dict[str, Any] | None:
    path = REPO_ROOT / "filin" / "ml" / "artifacts" / f"baseline_v0_1_{train}_to_{test}" / "model_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Сводный отчёт Docker multi-run эксперимента.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [run_info(name) for name in args.runs]
    pairs = [(args.runs[0], args.runs[1]), (args.runs[1], args.runs[2]), (args.runs[0], args.runs[2])] if len(args.runs) >= 3 else []
    lines = ["# Филин v0.2.1 — Docker multi-run experiment", "", "## Цель эксперимента", "", "Оценить воспроизводимость лабораторного pipeline и переносимость baseline-моделей между разными Docker-прогонами.", "", "## Конфигурация прогонов", ""]
    for run in runs:
        manifest = run["manifest"]
        lines.append(f"- `{run['name']}`: time_scale={manifest.get('time_scale')}, random_seed={manifest.get('random_seed')}, repeat={manifest.get('repeat')}, режим={manifest.get('execution_mode')}")
    lines.extend(["", "## Происхождение данных", "", "События получены при реальном выполнении действий между контейнерами и наблюдались со стороны traffic-client. Это не независимые сетевые flow.", "", "## Объёмы datasets", "", "| Run | Windows | Flows | Распределение классов |", "| --- | ---: | ---: | --- |"])
    lines.extend(f"| {run['name']} | {run['windows_rows']} | {run['flows_rows']} | {run['classes']} |" for run in runs)
    lines.extend(["", "## Docker-to-Docker эксперименты", "", "| Experiment | Train | Test | Best model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for train, test in pairs:
        metadata = experiment_metadata(train, test)
        if not metadata:
            lines.append(f"| {train}->{test} | {train} | {test} | нет данных | - | - | - | - |")
            continue
        metrics = metadata["metrics"]
        lines.append(f"| {train}->{test} | {train} | {test} | {metadata['model_name']} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |")
    multi_path = REPO_ROOT / "filin" / "ml" / "artifacts" / "docker_001_002_to_003" / "model_metadata.json"
    multi = load_json(multi_path)
    lines.extend(["", "## Обучение на двух runs и оценка на третьем", ""])
    if multi:
        metrics = multi["metrics"]
        lines.append(f"- Train: `{args.runs[0]} + {args.runs[1]}`; test: `{args.runs[2]}`; лучшая модель: `{multi['model_name']}`; accuracy={metrics['accuracy']:.4f}; balanced_accuracy={metrics['balanced_accuracy']:.4f}; macro_f1={metrics['macro_f1']:.4f}; weighted_f1={metrics['weighted_f1']:.4f}.")
    else:
        lines.append("- Данные объединённого эксперимента пока не сформированы.")
    lines.extend(["", "## Метрики по классам", ""])
    for train, test in pairs:
        metadata = experiment_metadata(train, test)
        if not metadata:
            continue
        report = metadata["metrics"].get("classification_report", {})
        lines.extend([f"### {train} -> {test} ({metadata['model_name']})", "", "| Класс | Support | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"])
        for label, values in report.items():
            if isinstance(values, dict) and "support" in values:
                lines.append(f"| {label} | {values['support']:.0f} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1-score']:.4f} |")
        lines.extend(["", "Confusion matrix сохранена в соответствующем отчёте baseline-моделей.", ""])
    lines.extend(["## Анализ feature drift", "", "| Feature | Comparison | PSI | Standardized mean difference | Zero-rate difference | Drift level |", "| --- | --- | ---: | ---: | ---: | --- |"])
    comparisons = [(args.runs[0], args.runs[1]), (args.runs[1], args.runs[2]), (args.runs[0], args.runs[2]), ("mock", args.runs[0])]
    for source, target in comparisons:
        name = "drift_mock_to_docker.json" if source == "mock" else f"drift_{source.removeprefix('run_')}_to_{target.removeprefix('run_')}.json"
        data = load_json(REPO_ROOT / "filin" / "ml" / "reports" / name)
        if not data:
            continue
        for feature in data.get("features", [])[:3]:
            lines.append(f"| {feature['feature']} | {source}->{target} | {feature['population_stability_index']:.4f} | {feature['standardized_mean_difference']:.4f} | {feature['zero_rate_difference']:.4f} | {feature['drift_level']} |")
    lines.extend(["", "## Сравнение mock и Docker", "", "Сравнение mock -> Docker отражает смену источника данных, а не оценку на производственном трафике.", "", "## Ограничения", "", "Эксперименты Docker-to-Docker оценивают воспроизводимость лабораторного pipeline и переносимость моделей между разными прогонами одного стенда. Они не являются подтверждением качества на производственном сетевом трафике.", "", "## Следующий этап", "", "Подключение Zeek/Suricata для независимого наблюдения conn/http/dns flow и расширение объёма разнородных лабораторных данных.", ""])
    output = Path(args.output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Сводный отчёт: {output}")


if __name__ == "__main__":
    main()
