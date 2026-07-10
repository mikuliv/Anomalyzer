from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = REPO_ROOT / "filin" / "ml" / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from build_flows_dataset import build_flows_dataset  # noqa: E402
from build_windows_dataset import build_windows_dataset  # noqa: E402
from dataset_report import build_report, read_jsonl, read_manifest  # noqa: E402
from normalize_events import normalize_events  # noqa: E402
from scenario_runner import (  # noqa: E402
    discover_scenario_paths,
    execute_manifest,
    parse_allowed_targets,
    parse_base_time,
    run_scenarios,
)
from label_writer import create_empty_manifest, save_manifest  # noqa: E402


def dataset_output_paths(run_dir: Path) -> tuple[Path, Path]:
    run_name = run_dir.name
    datasets_dir = REPO_ROOT / "filin" / "lab" / "output" / "datasets"
    return datasets_dir / f"windows_v0_1_{run_name}.csv", datasets_dir / f"flows_v0_1_{run_name}.csv"


def write_report(run_dir: Path) -> Path:
    report_path = run_dir / "dataset_report.md"
    report = build_report(
        manifest=read_manifest(run_dir / "scenario_manifest.yaml"),
        execution_events=read_jsonl(run_dir / "execution_events.jsonl"),
        traffic_events=read_jsonl(run_dir / "traffic_events.jsonl"),
        normalized_events=read_jsonl(run_dir / "normalized_events.jsonl"),
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "scenario_manifest.yaml"
    save_manifest(
        manifest_path,
        create_empty_manifest(
            dry_run=True,
            schedule_mode=args.schedule_mode,
            gap_seconds=args.gap_seconds,
            repeat=args.repeat,
        ),
    )
    scenario_paths = discover_scenario_paths(
        None,
        Path(args.scenarios),
        schedule_mode=args.schedule_mode,
        repeat=args.repeat,
    )
    success_count, error_count = run_scenarios(
        paths=scenario_paths,
        manifest_path=manifest_path,
        allowed_targets=parse_allowed_targets(None),
        dry_run=True,
        stop_on_error=True,
        base_time=parse_base_time(args.base_time),
        schedule_mode=args.schedule_mode,
        gap_seconds=args.gap_seconds,
        repeat=args.repeat,
    )
    if error_count:
        raise RuntimeError(f"Dry-run завершился с ошибками: {error_count}")
    print(f"Manifest создан: {manifest_path}")
    print(f"Сценариев проверено: {success_count}")

    completed, failed, skipped = execute_manifest(
        manifest_path=manifest_path,
        allow_dry_run_manifest=True,
        respect_schedule=False,
        max_runtime_seconds=args.max_runtime_seconds,
        mock=args.mock,
    )
    print(f"Выполнение: completed={completed}, failed={failed}, skipped={skipped}")
    if failed:
        raise RuntimeError(f"Выполнение сценариев завершилось с ошибками: {failed}")

    normalized_path = run_dir / "normalized_events.jsonl"
    normalize_events(run_dir / "execution_events.jsonl", run_dir / "traffic_events.jsonl", normalized_path)
    print(f"Нормализованные события: {normalized_path}")

    report_path = write_report(run_dir)
    print(f"Отчёт: {report_path}")

    windows_path, flows_path = dataset_output_paths(run_dir)
    build_windows_dataset(manifest_path, normalized_path, windows_path, args.window_seconds)
    build_flows_dataset(manifest_path, normalized_path, flows_path)
    return {
        "manifest": manifest_path,
        "execution_events": run_dir / "execution_events.jsonl",
        "traffic_events": run_dir / "traffic_events.jsonl",
        "normalized_events": normalized_path,
        "dataset_report": report_path,
        "windows_dataset": windows_path,
        "flows_dataset": flows_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный запуск pipeline одного laboratory run Филин.")
    parser.add_argument("--run-dir", required=True, help="Папка артефактов laboratory run.")
    parser.add_argument("--scenarios", default="filin/lab/scenarios", help="Папка YAML-сценариев.")
    parser.add_argument("--base-time", required=True, help="Плановое начало первого сценария.")
    parser.add_argument("--schedule-mode", choices=("grouped", "natural"), default="natural", help="Режим расписания.")
    parser.add_argument("--gap-seconds", type=int, default=30, help="Пауза между сценариями.")
    parser.add_argument("--repeat", type=int, default=1, help="Количество повторов natural-последовательности.")
    parser.add_argument("--mock", action="store_true", help="Выполнить без сетевой активности.")
    parser.add_argument("--max-runtime-seconds", type=int, default=300, help="Общий лимит времени выполнения.")
    parser.add_argument("--window-seconds", type=int, default=60, help="Размер окна для window-level датасета.")
    args = parser.parse_args()

    outputs = run_pipeline(args)
    print("Созданные артефакты:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
