from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


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


def request_url(url: str, timeout: float) -> tuple[bool, int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(512).decode("utf-8", errors="replace")
            return True, int(response.status), body
    except urllib.error.HTTPError as error:
        body = error.read(512).decode("utf-8", errors="replace")
        return False, int(error.code), body
    except OSError as error:
        return False, None, str(error)


def check_services(checks: list[ServiceCheck], timeout: float) -> int:
    failures = 0
    for check in checks:
        ok, status, body = request_url(check.url, timeout)
        result = {
            "service": check.name,
            "url": check.url,
            "required": check.required,
            "ok": ok,
            "status": status,
            "response_preview": body[:160],
        }
        print(json.dumps(result, ensure_ascii=False))
        if not ok and check.required:
            failures += 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка доступности лабораторных сервисов Филин.")
    parser.add_argument("--mode", choices=("host", "docker"), required=True, help="Режим проверки сервисов.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Таймаут HTTP-проверки в секундах.")
    parser.add_argument("--control-url", default=None, help="Опциональный URL control-api для host-режима.")
    args = parser.parse_args()

    if args.mode == "host":
        checks = list(HOST_CHECKS)
        if args.control_url:
            checks.append(ServiceCheck("control-api", args.control_url, required=False))
    else:
        checks = list(DOCKER_CHECKS)

    failures = check_services(checks, timeout=args.timeout)
    if failures:
        raise SystemExit(f"Проверка завершилась с ошибками: {failures}")


if __name__ == "__main__":
    main()
