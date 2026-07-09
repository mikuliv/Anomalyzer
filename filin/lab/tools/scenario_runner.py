from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from label_writer import append_scenario_window, create_empty_manifest, save_manifest


DEFAULT_ALLOWED_TARGETS = {"target-web", "target-api", "control-api", "internal-dns"}
NATURAL_SCENARIO_ORDER = [
    "benign_api_usage",
    "benign_web_browsing",
    "attack_port_scan",
    "benign_dns_activity",
    "benign_file_downloads",
    "attack_auth_failures",
    "benign_ssh_admin",
    "attack_web_probe",
    "benign_web_browsing",
    "attack_low_rate_dos",
    "benign_api_usage",
    "attack_beacon_simulation",
    "benign_dns_activity",
]


def load_scenario(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Сценарий должен быть YAML-словарем.")
    return data


def collect_allowed_targets(scenario: dict[str, Any]) -> set[str]:
    targets = set(scenario.get("allowed_targets") or [])
    limits = scenario.get("safety_limits") or {}
    targets.update(limits.get("allowed_targets") or [])
    return targets


def validate_scenario(scenario: dict[str, Any], allowed_targets: set[str]) -> None:
    required_fields = {
        "scenario_id",
        "type",
        "name",
        "description",
        "source_role",
        "target_role",
        "expected_label",
        "duration_seconds",
        "intensity",
        "safety_limits",
        "notes",
    }
    missing = sorted(required_fields - set(scenario))
    if missing:
        raise ValueError(f"В сценарии отсутствуют обязательные поля: {', '.join(missing)}")

    limits = scenario.get("safety_limits") or {}
    external_allowed = bool(scenario.get("external_network_allowed", limits.get("external_network_allowed", False)))
    if external_allowed:
        raise ValueError("Сценарий запрещен: external_network_allowed должен быть false.")

    scenario_targets = collect_allowed_targets(scenario)
    if not scenario_targets:
        raise ValueError("В сценарии не задан allowlist целей.")

    unexpected_targets = sorted(scenario_targets - allowed_targets)
    if unexpected_targets:
        raise ValueError(f"Цели вне allowlist: {', '.join(unexpected_targets)}")

    target_role = scenario["target_role"]
    if target_role not in allowed_targets:
        raise ValueError(f"target_role не входит в allowlist: {target_role}")

    max_duration = int(scenario.get("max_duration_seconds", limits.get("max_duration_seconds", 0)))
    duration = int(scenario["duration_seconds"])
    if max_duration and duration > max_duration:
        raise ValueError("duration_seconds превышает max_duration_seconds.")


def parse_allowed_targets(raw: str | None) -> set[str]:
    if not raw:
        return set(DEFAULT_ALLOWED_TARGETS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_base_time(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC).replace(microsecond=0)
    normalized = raw.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scenario_sort_key(path: Path) -> tuple[int, str]:
    parts = {part.lower() for part in path.parts}
    if "benign" in parts:
        group = 0
    elif "attacks" in parts:
        group = 1
    else:
        group = 2
    return group, path.name.lower()


def discover_grouped_scenario_paths(scenarios_path: Path) -> list[Path]:
    return sorted(scenarios_path.rglob("*.yaml"), key=scenario_sort_key)


def discover_natural_scenario_paths(scenarios_path: Path, repeat: int) -> list[Path]:
    discovered = {
        load_scenario(path)["scenario_id"]: path
        for path in scenarios_path.rglob("*.yaml")
    }
    ordered_paths: list[Path] = []
    for _ in range(repeat):
        for scenario_id in NATURAL_SCENARIO_ORDER:
            path = discovered.get(scenario_id)
            if path is None:
                print(f"Предупреждение: сценарий из natural-расписания отсутствует: {scenario_id}")
                continue
            ordered_paths.append(path)
    return ordered_paths


def discover_scenario_paths(
    scenario_path: Path | None,
    scenarios_path: Path | None,
    schedule_mode: str,
    repeat: int,
) -> list[Path]:
    if scenario_path and scenarios_path:
        raise ValueError("Нельзя одновременно указывать --scenario и --scenarios.")
    if not scenario_path and not scenarios_path:
        raise ValueError("Нужно указать либо --scenario, либо --scenarios.")
    if repeat < 1:
        raise ValueError("repeat должен быть не меньше 1.")

    if scenario_path:
        if not scenario_path.is_file():
            raise ValueError(f"Файл сценария не найден: {scenario_path}")
        return [scenario_path]

    if scenarios_path is None or not scenarios_path.is_dir():
        raise ValueError(f"Папка сценариев не найдена: {scenarios_path}")
    if schedule_mode == "natural":
        return discover_natural_scenario_paths(scenarios_path, repeat)
    return discover_grouped_scenario_paths(scenarios_path)


def run_dry_scenario(
    scenario: dict[str, Any],
    manifest_path: Path,
    planned_start: datetime,
    run_sequence: int,
    schedule_mode: str,
    gap_seconds: int,
    repeat: int,
) -> datetime:
    duration = int(scenario["duration_seconds"])
    planned_finish = planned_start + timedelta(seconds=duration)
    append_scenario_window(
        manifest_path,
        scenario,
        run_sequence=run_sequence,
        planned_started_at=format_utc(planned_start),
        planned_finished_at=format_utc(planned_finish),
        dry_run=True,
        schedule_mode=schedule_mode,
        gap_seconds=gap_seconds,
        repeat=repeat,
    )
    print(f"Dry-run сценария выполнен: {scenario['scenario_id']}")
    return planned_finish


def run_scenarios(
    paths: list[Path],
    manifest_path: Path,
    allowed_targets: set[str],
    dry_run: bool,
    stop_on_error: bool,
    base_time: datetime,
    schedule_mode: str,
    gap_seconds: int,
    repeat: int,
) -> tuple[int, int]:
    success_count = 0
    error_count = 0
    planned_start = base_time

    for run_sequence, path in enumerate(paths, start=1):
        try:
            scenario = load_scenario(path)
            validate_scenario(scenario, allowed_targets)
            if not dry_run:
                raise ValueError("В v0.1 поддерживается только режим dry-run.")
            planned_finish = run_dry_scenario(
                scenario,
                manifest_path,
                planned_start,
                run_sequence=run_sequence,
                schedule_mode=schedule_mode,
                gap_seconds=gap_seconds,
                repeat=repeat,
            )
            planned_start = planned_finish + timedelta(seconds=gap_seconds)
            success_count += 1
        except Exception as error:
            error_count += 1
            print(f"Ошибка сценария {path}: {error}")
            if stop_on_error:
                break

    return success_count, error_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Безопасный dry-run запуск YAML-сценариев Филин.")
    parser.add_argument("--scenario", default=None, help="Путь к одному YAML-сценарию.")
    parser.add_argument("--scenarios", default=None, help="Путь к папке YAML-сценариев.")
    parser.add_argument("--manifest", required=True, help="Путь к manifest разметки.")
    parser.add_argument("--allowed-targets", default=None, help="Список разрешенных целей через запятую.")
    parser.add_argument("--dry-run", action="store_true", help="Проверить сценарии без генерации трафика.")
    parser.add_argument("--stop-on-error", action="store_true", help="Остановиться при первой ошибке.")
    parser.add_argument("--reset-manifest", action="store_true", help="Удалить существующий manifest перед запуском.")
    parser.add_argument("--base-time", default=None, help="Начало первого сценария, например 2026-07-09T13:00:00Z.")
    parser.add_argument(
        "--schedule-mode",
        choices=("grouped", "natural"),
        default="grouped",
        help="Режим расписания: grouped для проверки, natural для естественного датасета.",
    )
    parser.add_argument("--gap-seconds", type=int, default=0, help="Пауза между плановыми окнами сценариев.")
    parser.add_argument("--repeat", type=int, default=1, help="Количество повторов natural-последовательности.")
    args = parser.parse_args()

    if args.gap_seconds < 0:
        raise ValueError("gap-seconds не может быть отрицательным.")

    manifest_path = Path(args.manifest)
    if args.reset_manifest and manifest_path.exists():
        manifest_path.unlink()
    if args.reset_manifest:
        save_manifest(
            manifest_path,
            create_empty_manifest(
                dry_run=args.dry_run,
                schedule_mode=args.schedule_mode,
                gap_seconds=args.gap_seconds,
                repeat=args.repeat,
            ),
        )

    paths = discover_scenario_paths(
        Path(args.scenario) if args.scenario else None,
        Path(args.scenarios) if args.scenarios else None,
        schedule_mode=args.schedule_mode,
        repeat=args.repeat,
    )
    allowed_targets = parse_allowed_targets(args.allowed_targets)
    success_count, error_count = run_scenarios(
        paths=paths,
        manifest_path=manifest_path,
        allowed_targets=allowed_targets,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        base_time=parse_base_time(args.base_time),
        schedule_mode=args.schedule_mode,
        gap_seconds=args.gap_seconds,
        repeat=args.repeat,
    )

    print("Итог dry-run:")
    print(f"- YAML-файлов найдено: {len(paths)}")
    print(f"- Сценариев успешно проверено: {success_count}")
    print(f"- Сценариев с ошибкой: {error_count}")
    print(f"- Manifest: {manifest_path}")

    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
