from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from typing import Callable

import requests


ALLOWED_HTTP_TARGETS = {
    "target-web": "http://target-web",
    "target-api": "http://target-api:8080",
    "control-api": "http://control-api:8090",
}
ALLOWED_PORTS = {
    "target-web": [80],
    "target-api": [8080],
    "control-api": [8090],
}


@dataclass(frozen=True)
class ClientConfig:
    action: str
    target: str
    duration: int
    rate_per_minute: int
    timeout: float


def validate_config(config: ClientConfig) -> None:
    if config.target not in ALLOWED_HTTP_TARGETS:
        raise ValueError(f"Цель не входит во внутренний allowlist: {config.target}")
    if config.duration <= 0 or config.duration > 900:
        raise ValueError("Длительность должна быть от 1 до 900 секунд.")
    if config.rate_per_minute <= 0 or config.rate_per_minute > 60:
        raise ValueError("Интенсивность должна быть от 1 до 60 запросов в минуту.")


def sleep_limited(config: ClientConfig) -> None:
    delay = max(1.0, 60.0 / config.rate_per_minute)
    time.sleep(delay)


def bounded_loop(config: ClientConfig, action: Callable[[], None]) -> dict[str, int]:
    validate_config(config)
    deadline = time.time() + config.duration
    requests_sent = 0
    errors = 0
    while time.time() < deadline:
        try:
            action()
            requests_sent += 1
        except requests.RequestException:
            errors += 1
        sleep_limited(config)
    return {"requests_sent": requests_sent, "errors": errors}


def web_browsing(config: ClientConfig) -> dict[str, int]:
    paths = ["/", "/about.html", "/docs.html"]
    index = 0

    def action() -> None:
        nonlocal index
        requests.get(ALLOWED_HTTP_TARGETS["target-web"] + paths[index % len(paths)], timeout=config.timeout)
        index += 1

    return bounded_loop(config, action)


def api_usage(config: ClientConfig) -> dict[str, int]:
    paths = ["/health", "/api/items", "/api/status", "/api/profile/test-user"]
    index = 0

    def action() -> None:
        nonlocal index
        requests.get(ALLOWED_HTTP_TARGETS["target-api"] + paths[index % len(paths)], timeout=config.timeout)
        index += 1

    return bounded_loop(config, action)


def file_downloads(config: ClientConfig) -> dict[str, int]:
    paths = ["/files/sample-small.txt", "/files/sample-config.json"]
    index = 0

    def action() -> None:
        nonlocal index
        requests.get(ALLOWED_HTTP_TARGETS["target-web"] + paths[index % len(paths)], timeout=config.timeout)
        index += 1

    return bounded_loop(config, action)


def auth_failures(config: ClientConfig) -> dict[str, int]:
    def action() -> None:
        requests.post(
            ALLOWED_HTTP_TARGETS["target-api"] + "/api/login",
            json={"username": "test-user", "password": "wrong-test-password"},
            timeout=config.timeout,
        )

    return bounded_loop(config, action)


def web_probe(config: ClientConfig) -> dict[str, int]:
    paths = ["/admin-test", "/debug-test", "/backup-test", "/old-login-test", "/not-found-test"]
    index = 0

    def action() -> None:
        nonlocal index
        requests.get(ALLOWED_HTTP_TARGETS["target-web"] + paths[index % len(paths)], timeout=config.timeout)
        index += 1

    return bounded_loop(config, action)


def low_rate_dos(config: ClientConfig) -> dict[str, int]:
    if config.rate_per_minute > 15:
        raise ValueError("Для low_rate_dos разрешено не более 15 запросов в минуту.")
    return web_browsing(config)


def beacon_simulation(config: ClientConfig) -> dict[str, int]:
    def action() -> None:
        requests.post(
            ALLOWED_HTTP_TARGETS["control-api"] + "/beacon",
            json={"source": "traffic-client", "mode": "учебный heartbeat"},
            timeout=config.timeout,
        )

    return bounded_loop(config, action)


def port_scan(config: ClientConfig) -> dict[str, int]:
    validate_config(config)
    if config.target not in ALLOWED_PORTS:
        raise ValueError(f"Для цели не задан allowlist портов: {config.target}")
    checks = 0
    errors = 0
    for port in ALLOWED_PORTS[config.target]:
        try:
            with socket.create_connection((config.target, port), timeout=config.timeout):
                checks += 1
        except OSError:
            errors += 1
    return {"requests_sent": checks, "errors": errors}


ACTION_HANDLERS = {
    "web_browsing": web_browsing,
    "api_usage": api_usage,
    "file_downloads": file_downloads,
    "auth_failures": auth_failures,
    "web_probe": web_probe,
    "low_rate_dos": low_rate_dos,
    "beacon_simulation": beacon_simulation,
    "port_scan": port_scan,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Безопасный traffic-client для лабораторного стенда Филин.")
    parser.add_argument("--action", required=True, choices=sorted(ACTION_HANDLERS))
    parser.add_argument("--target", required=True, choices=sorted(ALLOWED_HTTP_TARGETS))
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rate-per-minute", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    config = ClientConfig(
        action=args.action,
        target=args.target,
        duration=args.duration,
        rate_per_minute=args.rate_per_minute,
        timeout=args.timeout,
    )
    result = ACTION_HANDLERS[args.action](config)
    print(result)


if __name__ == "__main__":
    main()
