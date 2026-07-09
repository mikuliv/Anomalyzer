from __future__ import annotations

import argparse
from dataclasses import dataclass


ALLOWED_TARGETS = {"target-web", "target-api", "control-api", "internal-dns"}


@dataclass(frozen=True)
class TrafficPlan:
    target: str
    duration: int
    rate_per_minute: int
    dry_run: bool


def validate_plan(plan: TrafficPlan) -> None:
    if plan.target not in ALLOWED_TARGETS:
        raise ValueError(f"Цель не входит во внутренний allowlist: {plan.target}")
    if plan.duration <= 0 or plan.duration > 3600:
        raise ValueError("Длительность должна быть в диапазоне от 1 до 3600 секунд.")
    if plan.rate_per_minute <= 0 or plan.rate_per_minute > 60:
        raise ValueError("Интенсивность должна быть в диапазоне от 1 до 60 запросов в минуту.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор обычного безопасного трафика для стенда Филин.")
    parser.add_argument("--target", required=True, help="Внутренняя цель из allowlist.")
    parser.add_argument("--duration", type=int, default=60, help="Длительность в секундах.")
    parser.add_argument("--rate-per-minute", type=int, default=5, help="Лимит запросов в минуту.")
    parser.add_argument("--dry-run", action="store_true", help="Показать план без сетевых обращений.")
    args = parser.parse_args()

    plan = TrafficPlan(
        target=args.target,
        duration=args.duration,
        rate_per_minute=args.rate_per_minute,
        dry_run=args.dry_run,
    )
    validate_plan(plan)

    if not plan.dry_run:
        raise ValueError("В v0.1 генератор работает только в dry-run, чтобы исключить случайные внешние обращения.")

    print(
        "План обычного трафика: "
        f"цель={plan.target}, длительность={plan.duration}, "
        f"интенсивность={plan.rate_per_minute}/мин."
    )


if __name__ == "__main__":
    main()
