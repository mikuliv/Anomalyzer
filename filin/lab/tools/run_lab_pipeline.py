from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = REPO_ROOT / "filin" / "ml" / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from build_flows_dataset import build_flows_dataset  # noqa: E402
from build_windows_dataset import build_windows_dataset  # noqa: E402
from collect_service_logs import collect_logs  # noqa: E402
from dataset_report import build_report, read_jsonl, read_manifest  # noqa: E402
from normalize_events import normalize_events  # noqa: E402
from scenario_runner import discover_scenario_paths, execute_manifest, parse_allowed_targets, parse_base_time, run_scenarios  # noqa: E402
from label_writer import create_empty_manifest, save_manifest  # noqa: E402


LAB_SERVICES = ["target-web", "target-api", "control-api", "target-ssh-sim", "traffic-client"]


def dataset_output_paths(run_dir: Path) -> tuple[Path, Path]:
    datasets_dir = REPO_ROOT / "filin" / "lab" / "output" / "datasets"
    return datasets_dir / f"windows_v0_1_{run_dir.name}.csv", datasets_dir / f"flows_v0_1_{run_dir.name}.csv"


def docker_command(compose_file: Path, *arguments: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), *arguments]


def run_docker(command: list[str], project_dir: Path, failure_message: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, encoding="utf-8", check=False)
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{failure_message}: {details}")
    return completed


def start_services(args: argparse.Namespace, compose_file: Path, project_dir: Path) -> None:
    command = docker_command(compose_file, "up", "-d")
    if args.rebuild_services:
        command.append("--build")
    command.extend(LAB_SERVICES)
    run_docker(command, project_dir, "Не удалось запустить сервисы Docker-стенда")


def preflight(compose_file: Path, project_dir: Path) -> None:
    command = docker_command(
        compose_file, "exec", "-T", "traffic-client", "python", "/workspace/lab/tools/check_lab_services.py", "--mode", "docker",
    )
    completed: subprocess.CompletedProcess[str] | None = None
    for _attempt in range(10):
        completed = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, encoding="utf-8", check=False)
        if completed.returncode == 0:
            break
        time.sleep(1)
    assert completed is not None
    ssh_check = docker_command(compose_file, "exec", "-T", "traffic-client", "python", "-c", "import socket; s=socket.create_connection(('target-ssh-sim', 2222), 2); print(s.recv(64).decode().strip()); s.close()")
    ssh_completed = subprocess.run(ssh_check, cwd=project_dir, capture_output=True, text=True, encoding="utf-8", check=False)
    if completed.returncode or ssh_completed.returncode:
        details = "\n".join(value for value in [completed.stderr.strip(), completed.stdout.strip(), ssh_completed.stderr.strip()] if value)
        raise RuntimeError(
            "Не все обязательные сервисы Docker-стенда доступны. "
            "Запустите стенд командой: cd H:\\Anomalyzer\\filin\\lab\\docker; "
            "docker compose -f docker-compose.lab.yml up -d --build. " + details
        )
    print("Предварительная проверка Docker-стенда пройдена.")


def write_report(run_dir: Path) -> Path:
    report_path = run_dir / "dataset_report.md"
    report_path.write_text(build_report(read_manifest(run_dir / "scenario_manifest.yaml"), read_jsonl(run_dir / "execution_events.jsonl"), read_jsonl(run_dir / "traffic_events.jsonl"), read_jsonl(run_dir / "normalized_events.jsonl"), run_dir / "service_logs"), encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace, campaign_metadata: dict[str, object] | None = None, scenario_variants: dict[int, dict[str, object]] | None = None, skip_legacy_dataset: bool = False) -> dict[str, Path]:
    if args.mock == args.docker:
        raise ValueError("Нужно указать ровно один режим: --mock или --docker.")
    if args.time_scale <= 0 or args.time_scale > 1:
        raise ValueError("time-scale должен быть больше 0 и не больше 1.")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "scenario_manifest.yaml"
    compose_file = Path(args.compose_file).resolve()
    project_dir = Path(args.compose_project_dir).resolve()

    if args.docker:
        if args.start_services or args.rebuild_services:
            start_services(args, compose_file, project_dir)
        preflight(compose_file, project_dir)

    manifest = create_empty_manifest(True, args.schedule_mode, args.gap_seconds, args.repeat)
    manifest.update({"execution_mode": "docker" if args.docker else "mock", "random_seed": args.random_seed, "time_scale": args.time_scale})
    if campaign_metadata:
        manifest.update(campaign_metadata)
    save_manifest(manifest_path, manifest)
    paths = discover_scenario_paths(None, Path(args.scenarios), args.schedule_mode, args.repeat)
    success_count, error_count = run_scenarios(paths, manifest_path, parse_allowed_targets(None), True, True, parse_base_time(args.base_time), args.schedule_mode, args.gap_seconds, args.repeat)
    if error_count:
        raise RuntimeError(f"Планирование сценариев завершилось с ошибками: {error_count}")
    if scenario_variants:
        from scenario_runner import load_manifest
        manifest = load_manifest(manifest_path)
        for scenario in manifest.get("scenarios", []):
            variant = scenario_variants.get(int(scenario["run_sequence"]), {})
            scenario.update(variant)
        save_manifest(manifest_path, manifest)
    completed, failed, skipped = execute_manifest(manifest_path, True, False, args.max_runtime_seconds, args.mock, compose_file if args.docker else None, project_dir if args.docker else None, args.time_scale, args.random_seed)
    if failed:
        raise RuntimeError(f"Выполнение сценариев завершилось с ошибками: {failed}")
    print(f"Сценариев проверено: {success_count}; выполнено: {completed}; пропущено: {skipped}")

    if args.docker:
        warnings = collect_logs(run_dir, compose_file, project_dir, args.base_time)
        for warning in warnings:
            print(f"Предупреждение: {warning}")
    normalized_path = run_dir / "normalized_events.jsonl"
    normalize_events(run_dir / "execution_events.jsonl", run_dir / "traffic_events.jsonl", normalized_path)
    report_path = write_report(run_dir)
    windows_path, flows_path = dataset_output_paths(run_dir)
    if not skip_legacy_dataset:
        build_windows_dataset(manifest_path, normalized_path, windows_path, args.window_seconds)
        build_flows_dataset(manifest_path, normalized_path, flows_path)
    if args.docker and args.stop_services_after_run:
        run_docker(docker_command(compose_file, "down"), project_dir, "Не удалось остановить Docker-стенд")
    result = {"manifest": manifest_path, "traffic_events": run_dir / "traffic_events.jsonl", "normalized_events": normalized_path, "dataset_report": report_path}
    if not skip_legacy_dataset:
        result.update({"windows_dataset": windows_path, "flows_dataset": flows_path})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный запуск одного laboratory run Филин.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scenarios", default="filin/lab/scenarios")
    parser.add_argument("--base-time", required=True)
    parser.add_argument("--schedule-mode", choices=("grouped", "natural"), default="natural")
    parser.add_argument("--gap-seconds", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Синтетические события без сетевой активности.")
    mode.add_argument("--docker", action="store_true", help="Реальные действия внутри изолированной Docker-сети.")
    parser.add_argument("--compose-file", default="filin/lab/docker/docker-compose.lab.yml")
    parser.add_argument("--compose-project-dir", default="filin/lab/docker")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--start-services", action="store_true")
    parser.add_argument("--rebuild-services", action="store_true")
    parser.add_argument("--stop-services-after-run", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=300)
    parser.add_argument("--window-seconds", type=int, default=60)
    args = parser.parse_args()
    for name, path in run_pipeline(args).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
