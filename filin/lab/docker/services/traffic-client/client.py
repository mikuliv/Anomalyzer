from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from datetime import UTC, datetime
from typing import Any

import requests


ALLOWED_HTTP_TARGETS = {
    "target-web": "http://target-web",
    "target-api": "http://target-api:8080",
    "control-api": "http://control-api:8090",
}
ALLOWED_TARGETS = set(ALLOWED_HTTP_TARGETS) | {"internal-dns", "target-ssh-sim"}
SCAN_PORTS = [22, 80, 443, 8000, 8080, 5601]
WEB_PATHS = ["/", "/about.html", "/docs.html", "/files/sample-small.txt", "/files/sample-config.json"]
API_PATHS = ["/health", "/api/items", "/api/status", "/api/profile/test-user"]
PROBE_PATHS = ["/admin-test", "/debug-test", "/backup-test", "/old-login-test", "/not-found-test"]
DNS_NAMES = ["target-web", "target-api", "control-api", "target-ssh-sim"]

SCENARIOS = {
    "benign_web_browsing": ("benign", "benign", "benign-client", "target-web"),
    "benign_api_usage": ("benign", "benign", "benign-client", "target-api"),
    "benign_dns_activity": ("benign", "benign", "benign-client", "internal-dns"),
    "benign_file_downloads": ("benign", "benign", "benign-client", "target-web"),
    "benign_ssh_admin": ("benign", "benign", "benign-client", "target-ssh-sim"),
    "attack_port_scan": ("attack", "port_scan", "attacker-simulator", "target-web"),
    "attack_auth_failures": ("attack", "auth_failures", "attacker-simulator", "target-api"),
    "attack_web_probe": ("attack", "web_probe", "attacker-simulator", "target-web"),
    "attack_low_rate_dos": ("attack", "low_rate_dos", "attacker-simulator", "target-web"),
    "attack_beacon_simulation": ("attack", "beacon_simulation", "attacker-simulator", "control-api"),
}


class SafetyError(ValueError):
    """Ошибка нарушения ограничений изолированного лабораторного стенда."""


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def event_base(args: argparse.Namespace, event_type: str, protocol: str, host: str, port: int | None) -> dict[str, Any]:
    scenario_type, label, source_role, target_role = SCENARIOS[args.scenario]
    return {
        "timestamp": now(),
        "run_id": args.run_id,
        "run_sequence": args.run_sequence,
        "scenario_id": args.scenario,
        "type": scenario_type,
        "label": label,
        "source_role": source_role,
        "target_role": target_role,
        "event_source": "traffic_client",
        "observation_source": "client",
        "execution_mode": "docker",
        "synthetic": False,
        "event_type": event_type,
        "protocol": protocol,
        "target_host": host,
        "target_port": port,
        "method": None,
        "path": None,
        "status": "ok",
        "status_code": None,
        "bytes_in": 0,
        "bytes_out": 0,
        "latency_ms": 0.0,
        "auth_success": None,
        "error": None,
        "details": {},
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.scenario not in SCENARIOS:
        raise KeyError(args.scenario)
    if args.duration_seconds <= 0 or args.duration_seconds > 60:
        raise SafetyError("Длительность сценария должна быть от 1 до 60 секунд.")
    if args.max_events <= 0 or args.max_events > 100:
        raise SafetyError("max-events должен быть от 1 до 100.")
    if args.max_rate <= 0 or args.max_rate > 5:
        raise SafetyError("max-rate должен быть от 1 до 5 действий в секунду.")
    if args.output_format != "jsonl":
        raise SafetyError("Поддерживается только формат вывода jsonl.")


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def request_event(args: argparse.Namespace, method: str, target: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if target not in ALLOWED_HTTP_TARGETS:
        raise SafetyError(f"HTTP-цель не входит в allowlist: {target}")
    if not path.startswith("/") or "://" in path or "?" in path:
        raise SafetyError("Разрешены только фиксированные внутренние пути без query-параметров.")
    url = ALLOWED_HTTP_TARGETS[target] + path
    event = event_base(args, "http_request", "http", target, 80 if target == "target-web" else (8080 if target == "target-api" else 8090))
    event["method"] = method
    event["path"] = path
    started = time.perf_counter()
    try:
        response = requests.request(method, url, json=payload, timeout=2.0, allow_redirects=False)
        event["status_code"] = response.status_code
        event["bytes_in"] = len(response.content)
        event["bytes_out"] = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) if payload else 0
        event["status"] = "ok" if response.status_code < 500 else "error"
        event["details"] = {"content_type": response.headers.get("content-type", "")}
    except requests.RequestException as error:
        event["status"] = "error"
        event["error"] = str(error)
    event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return event


def tcp_event(args: argparse.Namespace, host: str, port: int, event_type: str) -> dict[str, Any]:
    if host not in ALLOWED_TARGETS:
        raise SafetyError(f"TCP-цель не входит в allowlist: {host}")
    event = event_base(args, event_type, "tcp", host, port)
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=1.0) as connection:
            if event_type == "admin_tcp_session_check":
                event["bytes_in"] = len(connection.recv(64))
        event["status"] = "open"
    except socket.timeout:
        event["status"] = "timeout"
    except OSError as error:
        event["status"] = "closed"
        event["error"] = str(error)
    event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return event


def dns_event(args: argparse.Namespace, name: str) -> dict[str, Any]:
    if name not in DNS_NAMES:
        raise SafetyError(f"DNS-имя не входит в allowlist: {name}")
    event = event_base(args, "dns_resolution", "dns", name, 53)
    started = time.perf_counter()
    try:
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(name, None)})
        event["details"] = {"resolved_addresses": addresses, "scope": "internal"}
    except OSError as error:
        event["status"] = "error"
        event["error"] = str(error)
    event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return event


def scenario_events(args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    capacity = max(1, min(args.max_events, int(args.duration_seconds * args.max_rate)))
    if args.scenario == "benign_web_browsing":
        return [request_event(args, "GET", "target-web", rng.choice(WEB_PATHS)) for _ in range(min(80, capacity))]
    if args.scenario == "benign_api_usage":
        return [request_event(args, "GET", "target-api", rng.choice(API_PATHS)) for _ in range(min(90, capacity))]
    if args.scenario == "benign_dns_activity":
        return [dns_event(args, rng.choice(DNS_NAMES)) for _ in range(min(60, capacity))]
    if args.scenario == "benign_file_downloads":
        paths = ["/files/sample-small.txt", "/files/sample-config.json"]
        return [request_event(args, "GET", "target-web", rng.choice(paths)) for _ in range(min(20, capacity))]
    if args.scenario == "benign_ssh_admin":
        return [tcp_event(args, "target-ssh-sim", 2222, "admin_tcp_session_check") for _ in range(min(15, capacity))]
    if args.scenario == "attack_port_scan":
        return [tcp_event(args, "target-web", rng.choice(SCAN_PORTS), "tcp_connect_check") for _ in range(min(60, capacity))]
    if args.scenario == "attack_auth_failures":
        events = []
        for _ in range(min(30, capacity)):
            event = request_event(args, "POST", "target-api", "/api/login", {"username": "test-user", "password": "invalid-lab-password"})
            event["event_type"] = "auth_attempt"
            event["auth_success"] = False
            event["details"] = {"username_kind": "test_user", "credential_kind": "invalid_lab_value"}
            events.append(event)
        return events
    if args.scenario == "attack_web_probe":
        return [request_event(args, "GET", "target-web", rng.choice(PROBE_PATHS)) for _ in range(min(30, capacity))]
    if args.scenario == "attack_low_rate_dos":
        return [request_event(args, "GET", "target-web", rng.choice(["/", "/about.html"])) for _ in range(min(100, capacity))]
    if args.scenario == "attack_beacon_simulation":
        events = []
        for number in range(1, min(60, capacity) + 1):
            event = request_event(args, "POST", "control-api", "/beacon", {"agent": "filin-lab-client", "status": "alive", "sequence": number})
            event["event_type"] = "heartbeat_request"
            event["details"] = {"pattern": "учебный heartbeat", "sequence": number}
            events.append(event)
        return events
    raise KeyError(args.scenario)


def main() -> None:
    parser = argparse.ArgumentParser(description="Безопасное выполнение одного лабораторного сценария внутри Docker-сети.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-sequence", required=True, type=int)
    parser.add_argument("--duration-seconds", required=True, type=int)
    parser.add_argument("--max-events", required=True, type=int)
    parser.add_argument("--max-rate", required=True, type=float)
    parser.add_argument("--random-seed", required=True, type=int)
    parser.add_argument("--output-format", default="jsonl")
    args = parser.parse_args()
    try:
        validate_args(args)
        rng = random.Random(args.random_seed)
        events = scenario_events(args, rng)
        delay = 1.0 / args.max_rate
        for index, event in enumerate(events):
            emit(event)
            if index + 1 < len(events):
                time.sleep(delay * rng.uniform(0.85, 1.15))
    except KeyError:
        print("Неизвестный лабораторный сценарий.", file=sys.stderr)
        raise SystemExit(3)
    except SafetyError as error:
        print(f"Нарушение ограничений безопасности: {error}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        print(f"Ошибка выполнения сценария: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
