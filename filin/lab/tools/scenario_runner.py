from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from label_writer import append_scenario_window, utc_now


DEFAULT_ALLOWED_TARGETS = {"target-web", "target-api", "control-api", "internal-dns"}


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


def run_dry_scenario(scenario: dict[str, Any], manifest_path: Path) -> None:
    started_at = utc_now()
    finished_at = utc_now()
    append_scenario_window(manifest_path, scenario, started_at, finished_at, dry_run=True)
    print(f"Dry-run сценария выполнен: {scenario['scenario_id']}")


def parse_allowed_targets(raw: str | None) -> set[str]:
    if not raw:
        return set(DEFAULT_ALLOWED_TARGETS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Безопасный dry-run запуск YAML-сценария Филин.")
    parser.add_argument("--scenario", required=True, help="Путь к YAML-сценарию.")
    parser.add_argument("--manifest", required=True, help="Путь к manifest разметки.")
    parser.add_argument("--allowed-targets", default=None, help="Список разрешенных целей через запятую.")
    parser.add_argument("--dry-run", action="store_true", help="Проверить сценарий без генерации трафика.")
    args = parser.parse_args()

    scenario = load_scenario(Path(args.scenario))
    allowed_targets = parse_allowed_targets(args.allowed_targets)
    validate_scenario(scenario, allowed_targets)

    if not args.dry_run:
        raise ValueError("В v0.1 поддерживается только режим dry-run.")

    run_dry_scenario(scenario, Path(args.manifest))


if __name__ == "__main__":
    main()
