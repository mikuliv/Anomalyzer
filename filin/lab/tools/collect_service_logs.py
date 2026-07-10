from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SERVICES = ["target-web", "target-api", "control-api", "traffic-client", "target-ssh-sim"]


def collect_logs(run_dir: Path, compose_file: Path, compose_project_dir: Path, since: str | None = None) -> list[str]:
    output_dir = run_dir / "service_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    for service in SERVICES:
        command = ["docker", "compose", "-f", str(compose_file), "logs", "--no-color"]
        if since:
            command.extend(["--since", since])
        command.append(service)
        completed = subprocess.run(command, cwd=compose_project_dir, capture_output=True, text=True, encoding="utf-8", check=False)
        path = output_dir / f"{service}.log"
        path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            warnings.append(f"Не удалось собрать лог {service}: {completed.stderr.strip() or 'сервис отсутствует'}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Сбор диагностических логов Docker-сервисов лабораторного стенда.")
    parser.add_argument("--run-dir", required=True, help="Папка артефактов laboratory run.")
    parser.add_argument("--compose-file", required=True, help="Путь к compose-файлу.")
    parser.add_argument("--compose-project-dir", default=None, help="Рабочая папка Docker Compose.")
    parser.add_argument("--since", default=None, help="Начальная временная отметка для docker compose logs.")
    args = parser.parse_args()
    warnings = collect_logs(Path(args.run_dir), Path(args.compose_file), Path(args.compose_project_dir or Path(args.compose_file).parent), args.since)
    print(f"Логи сохранены: {Path(args.run_dir) / 'service_logs'}")
    for warning in warnings:
        print(f"Предупреждение: {warning}")


if __name__ == "__main__":
    main()
