from __future__ import annotations

import argparse
from pathlib import Path

from train_baselines import train_baselines


REPO_ROOT = Path(__file__).resolve().parents[3]


def dataset_path(run_name: str) -> Path:
    return REPO_ROOT / "filin" / "lab" / "output" / "datasets" / f"windows_v0_1_{run_name}.csv"


def ensure_dataset(path: Path, run_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Датасет для {run_name} не найден: {path}. "
            f"Сначала выполните: python filin/lab/tools/run_lab_pipeline.py --run-dir filin/lab/output/runs/{run_name} "
            "--base-time 2026-07-09T13:00:00Z --mock --window-seconds 60"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск external ML эксперимента по двум laboratory runs.")
    parser.add_argument("--train-run", required=True, help="Имя train run, например run_001.")
    parser.add_argument("--test-run", required=True, help="Имя test run, например run_002.")
    parser.add_argument("--target", default="label", help="Целевая колонка.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Не используется в external mode, сохранено для совместимости.")
    parser.add_argument("--random-state", type=int, default=42, help="Фиксированное состояние генератора.")
    parser.add_argument("--min-class-count", type=int, default=2, help="Минимальное число объектов класса для предупреждения.")
    args = parser.parse_args()

    train_dataset = dataset_path(args.train_run)
    test_dataset = dataset_path(args.test_run)
    ensure_dataset(train_dataset, args.train_run)
    ensure_dataset(test_dataset, args.test_run)

    experiment_name = f"baseline_v0_1_{args.train_run}_to_{args.test_run}"
    output_dir = REPO_ROOT / "filin" / "ml" / "artifacts" / experiment_name
    report_path = REPO_ROOT / "filin" / "ml" / "reports" / f"{experiment_name}.md"
    result = train_baselines(
        dataset_path=train_dataset,
        external_test_dataset_path=test_dataset,
        target=args.target,
        output_dir=output_dir,
        report_path=report_path,
        test_size=args.test_size,
        random_state=args.random_state,
        min_class_count=args.min_class_count,
    )
    print(f"External experiment: {args.train_run} -> {args.test_run}")
    print(f"Лучшая модель: {result['best_model']}")
    print(f"Отчёт: {result['report_path']}")
    print(f"Модель: {result['model_path']}")


if __name__ == "__main__":
    main()
