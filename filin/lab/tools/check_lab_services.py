from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceCheck:
    name: str
    url: str
    required: bool = True


HOST_CHECKS = [
    ServiceCheck("target-web", "http://127.0.0.1:18080/"),
    ServiceCheck("target-api", "http://127.0.0.1:18081/health"),
    ServiceCheck("filin-backend", "http://127.0.0.1:8000/health", required=False),
]

DOCKER_CHECKS = [
    ServiceCheck("target-web", "http://target-web/"),
    ServiceCheck("target-api", "http://target-api:8080/health"),
    ServiceCheck("control-api", "http://control-api:8090/health"),
    ServiceCheck("filin-backend", "http://filin-backend:8000/health", required=False),
]


def request_url(url: str, timeout: float) -> tuple[bool, int | None, str, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(512).decode("utf-8", errors="replace")
            return True, int(response.status), body, None
    except urllib.error.HTTPError as error:
        body = error.read(512).decode("utf-8", errors="replace")
        return False, int(error.code), body, "http_error"
    except urllib.error.URLError as error:
        message = str(error.reason)
        return False, None, message, classify_network_error(message)
    except OSError as error:
        message = str(error)
        return False, None, message, classify_network_error(message)


def classify_network_error(message: str) -> str:
    lowered = message.lower()
    if "getaddrinfo failed" in lowered or "name or service not known" in lowered:
        return "dns_error"
    if "no address associated with hostname" in lowered:
        return "dns_error"
    if "nodename nor servname provided" in lowered:
        return "dns_error"
    if "connection refused" in lowered:
        return "connection_refused"
    return "network_error"


def check_services(checks: list[ServiceCheck], timeout: float, mode: str) -> tuple[int, int]:
    failures = 0
    dns_failures = 0
    for check in checks:
        ok, status, body, error_kind = request_url(check.url, timeout)
        result = {
            "service": check.name,
            "mode": mode,
            "url": check.url,
            "required": check.required,
            "ok": ok,
            "status": status,
            "error_kind": error_kind,
            "response_preview": body[:160],
        }
        print(json.dumps(result, ensure_ascii=False))
        if not ok and check.required:
            failures += 1
            if error_kind == "dns_error":
                dns_failures += 1
    return failures, dns_failures


def run_compose_exec(timeout: float) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    compose_file = repo_root / "filin" / "lab" / "docker" / "docker-compose.lab.yml"
    command = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "exec",
        "-T",
        "traffic-client",
        "python",
        "/workspace/lab/tools/check_lab_services.py",
        "--mode",
        "docker",
        "--timeout",
        str(timeout),
    ]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка доступности лабораторных сервисов Филин.")
    parser.add_argument(
        "--mode",
        choices=("host", "docker", "compose-exec"),
        required=True,
        help="Режим проверки сервисов.",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="Таймаут HTTP-проверки в секундах.")
    parser.add_argument("--control-url", default=None, help="Опциональный URL control-api для host-режима.")
    args = parser.parse_args()

    if args.mode == "host":
        checks = list(HOST_CHECKS)
        if args.control_url:
            checks.append(ServiceCheck("control-api", args.control_url, required=False))
    elif args.mode == "compose-exec":
        raise SystemExit(run_compose_exec(timeout=args.timeout))
    else:
        checks = list(DOCKER_CHECKS)

    failures, dns_failures = check_services(checks, timeout=args.timeout, mode=args.mode)
    if args.mode == "docker" and dns_failures:
        print(
            "Docker-режим должен выполняться внутри контейнера, подключенного к сети стенда, "
            "либо через docker compose exec traffic-client.",
            file=sys.stderr,
        )
    if failures:
        raise SystemExit(f"Проверка завершилась с ошибками: {failures}")


if __name__ == "__main__":
    main()
