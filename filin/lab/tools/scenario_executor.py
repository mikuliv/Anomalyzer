from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ALLOWED_TARGET_URLS = {
    "target-web": "http://target-web",
    "target-api": "http://target-api:8080",
    "control-api": "http://control-api:8090",
}
ALLOWED_PORTS = {
    "target-web": [80],
    "target-api": [8080],
    "control-api": [8090],
}
ACTION_BY_LABEL = {
    "benign": "web_browsing",
    "port_scan": "port_scan",
    "auth_failures": "auth_failures",
    "web_probe": "web_probe",
    "low_rate_dos": "low_rate_dos",
    "beacon_simulation": "beacon_simulation",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_event(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    action: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "run_id": manifest.get("run_id"),
        "run_sequence": scenario.get("run_sequence"),
        "scenario_id": scenario.get("scenario_id"),
        "type": scenario.get("type"),
        "label": scenario.get("label"),
        "source_role": scenario.get("source_role"),
        "target_role": scenario.get("target_role"),
        "action": action,
        "status": status,
        "details": details or {},
    }


def parse_rate(value: str | None, default: int = 5) -> int:
    if not value:
        return default
    number = value.split("/", 1)[0].strip()
    try:
        return max(1, int(number))
    except ValueError:
        return default


def bounded_loop(duration: int, rate_per_minute: int, action: Callable[[], None]) -> dict[str, int]:
    deadline = time.time() + duration
    delay = max(1.0, 60.0 / max(1, rate_per_minute))
    requests_sent = 0
    errors = 0
    while time.time() < deadline:
        try:
            action()
            requests_sent += 1
        except (OSError, urllib.error.URLError):
            errors += 1
        time.sleep(delay)
    return {"requests_sent": requests_sent, "errors": errors}


def safe_get(url: str, timeout: float = 2.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        response.read(256)


def safe_post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(256)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise


def validate_execution_target(scenario: dict[str, Any]) -> None:
    target = scenario.get("target_role")
    if target not in ALLOWED_TARGET_URLS:
        raise ValueError(f"Цель не входит во внутренний allowlist: {target}")


def duration_for_execution(scenario: dict[str, Any], mock: bool) -> int:
    if mock:
        return 0
    duration = int(scenario.get("duration_seconds", 1))
    return max(1, min(duration, 900))


def rate_for_execution(scenario: dict[str, Any]) -> int:
    safety = scenario.get("safety_limits") or {}
    return parse_rate(str(safety.get("max_rate", "")), default=5)


def execute_http_paths(scenario: dict[str, Any], paths: list[str], mock: bool) -> dict[str, Any]:
    validate_execution_target(scenario)
    if mock:
        return {"requests_sent": 0, "errors": 0, "mock": True}
    base_url = ALLOWED_TARGET_URLS[scenario["target_role"]]
    index = 0

    def action() -> None:
        nonlocal index
        safe_get(base_url + paths[index % len(paths)])
        index += 1

    return bounded_loop(duration_for_execution(scenario, mock), rate_for_execution(scenario), action)


def execute_api_usage(scenario: dict[str, Any], mock: bool) -> dict[str, Any]:
    return execute_http_paths(scenario, ["/health", "/api/items", "/api/status", "/api/profile/test-user"], mock)


def execute_auth_failures(scenario: dict[str, Any], mock: bool) -> dict[str, Any]:
    validate_execution_target(scenario)
    if mock:
        return {"requests_sent": 0, "errors": 0, "mock": True}
    base_url = ALLOWED_TARGET_URLS[scenario["target_role"]]

    def action() -> None:
        safe_post_json(
            base_url + "/api/login",
            {"username": "test-user", "password": "wrong-test-password"},
        )

    return bounded_loop(duration_for_execution(scenario, mock), rate_for_execution(scenario), action)


def execute_beacon(scenario: dict[str, Any], mock: bool) -> dict[str, Any]:
    validate_execution_target(scenario)
    if mock:
        return {"requests_sent": 0, "errors": 0, "mock": True}
    base_url = ALLOWED_TARGET_URLS[scenario["target_role"]]

    def action() -> None:
        safe_post_json(base_url + "/beacon", {"source": "scenario_executor", "mode": "учебный heartbeat"})

    return bounded_loop(duration_for_execution(scenario, mock), rate_for_execution(scenario), action)


def execute_port_scan(scenario: dict[str, Any], mock: bool) -> dict[str, Any]:
    validate_execution_target(scenario)
    target = scenario["target_role"]
    ports = ALLOWED_PORTS.get(target, [])
    if mock:
        return {"requests_sent": 0, "errors": 0, "mock": True, "ports": ports}
    checks = 0
    errors = 0
    for port in ports:
        try:
            with socket.create_connection((target, port), timeout=2.0):
                checks += 1
        except OSError:
            errors += 1
    return {"requests_sent": checks, "errors": errors, "ports": ports}


def execute_scenario(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    events_path: Path,
    mock: bool,
) -> dict[str, Any]:
    append_event(events_path, build_event(manifest, scenario, "scenario_started", "ok", {"mock": mock}))
    label = scenario.get("label")
    try:
        if label == "benign":
            if scenario.get("scenario_id") == "benign_dns_activity":
                details = {
                    "requests_sent": 0,
                    "errors": 0,
                    "mock": mock,
                    "note": "DNS-сценарий v0.1 фиксируется как окно разметки без отдельного DNS-сервиса.",
                }
            elif scenario.get("scenario_id") == "benign_api_usage":
                details = execute_api_usage(scenario, mock)
            elif scenario.get("scenario_id") == "benign_file_downloads":
                details = execute_http_paths(scenario, ["/files/sample-small.txt", "/files/sample-config.json"], mock)
            else:
                details = execute_http_paths(scenario, ["/", "/about.html", "/docs.html"], mock)
        elif label == "port_scan":
            details = execute_port_scan(scenario, mock)
        elif label == "auth_failures":
            details = execute_auth_failures(scenario, mock)
        elif label == "web_probe":
            details = execute_http_paths(
                scenario,
                ["/admin-test", "/debug-test", "/backup-test", "/old-login-test", "/not-found-test"],
                mock,
            )
        elif label == "low_rate_dos":
            if rate_for_execution(scenario) > 15:
                raise ValueError("Для low_rate_dos разрешено не более 15 запросов в минуту.")
            details = execute_http_paths(scenario, ["/", "/about.html"], mock)
        elif label == "beacon_simulation":
            details = execute_beacon(scenario, mock)
        else:
            raise ValueError(f"Неизвестная метка сценария: {label}")
    except Exception as error:
        details = {"errors": 1, "message": str(error), "mock": mock}
        append_event(events_path, build_event(manifest, scenario, "scenario_finished", "failed", details))
        return {"status": "failed", "details": details}

    append_event(events_path, build_event(manifest, scenario, "scenario_finished", "completed", details))
    return {"status": "completed", "details": details}
