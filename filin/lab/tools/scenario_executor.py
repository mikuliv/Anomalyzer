from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ALLOWED_TARGET_URLS = {
    "target-web": "http://target-web",
    "target-api": "http://target-api:8080",
    "control-api": "http://control-api:8090",
}
ALLOWED_PORTS = {
    "target-web": [22, 80, 443, 8000, 8080, 5601],
    "target-api": [8080],
    "control-api": [8090],
}
OPEN_PORTS = {
    "target-web": {80},
    "target-api": {8080},
    "control-api": {8090},
}
INTERNAL_DNS_NAMES = ["target-web", "target-api", "control-api", "internal-dns"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_execution_event(
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


def build_traffic_event(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    event_time: datetime,
    event_type: str,
    protocol: str,
    target_host: str,
    target_port: int | None,
    status: str,
    status_code: int | None = None,
    method: str | None = None,
    path: str | None = None,
    bytes_in: int = 0,
    bytes_out: int = 0,
    latency_ms: float | None = None,
    auth_success: bool | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": format_utc(event_time),
        "run_id": manifest.get("run_id"),
        "run_sequence": scenario.get("run_sequence"),
        "scenario_id": scenario.get("scenario_id"),
        "type": scenario.get("type"),
        "label": scenario.get("label"),
        "source_role": scenario.get("source_role"),
        "target_role": scenario.get("target_role"),
        "event_source": "traffic_client",
        "event_type": event_type,
        "protocol": protocol,
        "target_host": target_host,
        "target_port": target_port,
        "status": status,
        "status_code": status_code,
        "method": method,
        "path": path,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "latency_ms": round(latency_ms if latency_ms is not None else 0.0, 2),
        "auth_success": auth_success,
        "error": error,
        "details": details or {},
    }


def scenario_rng(scenario: dict[str, Any]) -> random.Random:
    seed = f"{scenario.get('scenario_id')}:{scenario.get('run_sequence')}"
    return random.Random(seed)


def planned_start(scenario: dict[str, Any]) -> datetime:
    raw = scenario.get("planned_started_at") or utc_now()
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def spread_times(scenario: dict[str, Any], count: int, jitter_seconds: int = 0) -> list[datetime]:
    start = planned_start(scenario)
    duration = max(1, int(scenario.get("duration_seconds") or 1))
    rng = scenario_rng(scenario)
    step = duration / max(1, count)
    times: list[datetime] = []
    for index in range(count):
        jitter = rng.randint(-jitter_seconds, jitter_seconds) if jitter_seconds else 0
        offset = max(0, int(index * step) + jitter)
        times.append(start + timedelta(seconds=min(offset, duration)))
    return times


def validate_target(target: str) -> None:
    if target not in ALLOWED_TARGET_URLS and target != "internal-dns":
        raise ValueError(f"Цель не входит во внутренний allowlist: {target}")


def safe_get(url: str, timeout: float = 2.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        response.read(256)
        return int(response.status)


def safe_post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> int:
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
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def maybe_perform_http(event: dict[str, Any], mock: bool) -> dict[str, Any]:
    if mock:
        return event
    target = event["target_host"]
    validate_target(target)
    if target not in ALLOWED_TARGET_URLS:
        return event
    url = ALLOWED_TARGET_URLS[target] + (event.get("path") or "/")
    started = time.perf_counter()
    try:
        if event.get("method") == "POST":
            status_code = safe_post_json(url, {"source": "scenario_executor"})
        else:
            status_code = safe_get(url)
        event["status"] = "ok" if status_code < 500 else "error"
        event["status_code"] = status_code
        event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    except (OSError, urllib.error.URLError) as error:
        event["status"] = "error"
        event["error"] = str(error)
        event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return event


def http_events(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    count: int,
    target: str,
    paths: list[str],
    status_codes: list[int],
    bytes_range: tuple[int, int],
    method: str = "GET",
    mock: bool = True,
) -> list[dict[str, Any]]:
    validate_target(target)
    rng = scenario_rng(scenario)
    events: list[dict[str, Any]] = []
    for index, event_time in enumerate(spread_times(scenario, count)):
        status_code = status_codes[index % len(status_codes)]
        event = build_traffic_event(
            manifest,
            scenario,
            event_time,
            event_type="http_request",
            protocol="http",
            target_host=target,
            target_port=80 if target == "target-web" else 8080,
            status="ok" if status_code < 500 else "error",
            status_code=status_code,
            method=method,
            path=paths[index % len(paths)],
            bytes_in=rng.randint(*bytes_range),
            bytes_out=rng.randint(220, 520),
            latency_ms=rng.uniform(12, 55),
        )
        events.append(maybe_perform_http(event, mock))
    return events


def dns_events(manifest: dict[str, Any], scenario: dict[str, Any]) -> list[dict[str, Any]]:
    rng = scenario_rng(scenario)
    count = 34
    events: list[dict[str, Any]] = []
    for index, event_time in enumerate(spread_times(scenario, count)):
        query = INTERNAL_DNS_NAMES[index % len(INTERNAL_DNS_NAMES)]
        events.append(
            build_traffic_event(
                manifest,
                scenario,
                event_time,
                event_type="dns_query",
                protocol="dns",
                target_host="internal-dns",
                target_port=53,
                status="ok",
                bytes_in=rng.randint(80, 180),
                bytes_out=rng.randint(50, 120),
                latency_ms=rng.uniform(2, 12),
                details={"query": query, "scope": "internal"},
            )
        )
    return events


def admin_session_events(manifest: dict[str, Any], scenario: dict[str, Any]) -> list[dict[str, Any]]:
    rng = scenario_rng(scenario)
    phases = ["session_opened", "config_viewed", "status_checked", "session_closed"]
    events: list[dict[str, Any]] = []
    for index, event_time in enumerate(spread_times(scenario, 8)):
        events.append(
            build_traffic_event(
                manifest,
                scenario,
                event_time,
                event_type="admin_session_event",
                protocol="ssh",
                target_host="target-api",
                target_port=22,
                status="ok",
                bytes_in=rng.randint(120, 600),
                bytes_out=rng.randint(90, 300),
                latency_ms=rng.uniform(5, 30),
                details={"phase": phases[index % len(phases)], "credential_kind": "test_window"},
            )
        )
    return events


def port_scan_events(manifest: dict[str, Any], scenario: dict[str, Any], mock: bool) -> list[dict[str, Any]]:
    rng = scenario_rng(scenario)
    target = scenario.get("target_role", "target-web")
    validate_target(target)
    ports = ALLOWED_PORTS[target]
    events: list[dict[str, Any]] = []
    for index, event_time in enumerate(spread_times(scenario, 36)):
        port = ports[index % len(ports)]
        is_open = port in OPEN_PORTS.get(target, set())
        status = "ok" if is_open else "closed"
        latency = rng.uniform(4, 22)
        if not mock:
            started = time.perf_counter()
            try:
                with socket.create_connection((target, port), timeout=2.0):
                    status = "ok"
            except OSError:
                status = "closed"
            latency = (time.perf_counter() - started) * 1000
        events.append(
            build_traffic_event(
                manifest,
                scenario,
                event_time,
                event_type="tcp_connect_check",
                protocol="tcp",
                target_host=target,
                target_port=port,
                status=status,
                bytes_in=0,
                bytes_out=0,
                latency_ms=latency,
                details={"scan_mode": "allowlist_tcp_connect"},
            )
        )
    return events


def auth_events(manifest: dict[str, Any], scenario: dict[str, Any], mock: bool) -> list[dict[str, Any]]:
    rng = scenario_rng(scenario)
    events: list[dict[str, Any]] = []
    for event_time in spread_times(scenario, 24):
        event = build_traffic_event(
            manifest,
            scenario,
            event_time,
            event_type="auth_attempt",
            protocol="http",
            target_host="target-api",
            target_port=8080,
            status="ok",
            status_code=401,
            method="POST",
            path="/api/login",
            bytes_in=rng.randint(120, 260),
            bytes_out=rng.randint(180, 360),
            latency_ms=rng.uniform(16, 45),
            auth_success=False,
            details={"username_kind": "test_user"},
        )
        events.append(maybe_perform_http(event, mock))
    return events


def beacon_events(manifest: dict[str, Any], scenario: dict[str, Any], mock: bool) -> list[dict[str, Any]]:
    rng = scenario_rng(scenario)
    events: list[dict[str, Any]] = []
    for index, event_time in enumerate(spread_times(scenario, 32, jitter_seconds=3)):
        method = "POST" if index % 3 == 0 else "GET"
        event = build_traffic_event(
            manifest,
            scenario,
            event_time,
            event_type="heartbeat_request",
            protocol="http",
            target_host="control-api",
            target_port=8090,
            status="ok",
            status_code=200,
            method=method,
            path="/beacon",
            bytes_in=rng.randint(80, 180),
            bytes_out=rng.randint(80, 180),
            latency_ms=rng.uniform(10, 32),
            details={"pattern": "почти регулярный heartbeat", "jitter_seconds": 3},
        )
        events.append(maybe_perform_http(event, mock))
    return events


def generate_traffic_events(manifest: dict[str, Any], scenario: dict[str, Any], mock: bool) -> list[dict[str, Any]]:
    scenario_id = scenario.get("scenario_id")
    label = scenario.get("label")
    if scenario_id == "benign_api_usage":
        return http_events(
            manifest,
            scenario,
            48,
            "target-api",
            ["/api/items", "/api/status", "/api/profile/test-user", "/api/missing-test"],
            [200, 200, 200, 404],
            (500, 1800),
            mock=mock,
        )
    if scenario_id == "benign_web_browsing":
        return http_events(
            manifest,
            scenario,
            50,
            "target-web",
            ["/", "/about.html", "/docs.html", "/files/sample-small.txt", "/files/sample-config.json"],
            [200, 200, 200, 200, 200],
            (900, 2600),
            mock=mock,
        )
    if scenario_id == "benign_dns_activity":
        return dns_events(manifest, scenario)
    if scenario_id == "benign_file_downloads":
        return http_events(
            manifest,
            scenario,
            12,
            "target-web",
            ["/files/sample-small.txt", "/files/sample-config.json"],
            [200, 200],
            (1800, 4800),
            mock=mock,
        )
    if scenario_id == "benign_ssh_admin":
        return admin_session_events(manifest, scenario)
    if label == "port_scan":
        return port_scan_events(manifest, scenario, mock)
    if label == "auth_failures":
        return auth_events(manifest, scenario, mock)
    if label == "web_probe":
        return http_events(
            manifest,
            scenario,
            18,
            "target-web",
            ["/admin-test", "/debug-test", "/backup-test", "/old-login-test", "/not-found-test"],
            [404, 403, 404, 404, 404],
            (180, 650),
            mock=mock,
        )
    if label == "low_rate_dos":
        return http_events(
            manifest,
            scenario,
            60,
            "target-web",
            ["/", "/about.html"],
            [200, 200],
            (700, 1800),
            mock=mock,
        )
    if label == "beacon_simulation":
        return beacon_events(manifest, scenario, mock)
    raise ValueError(f"Неизвестный сценарий для генерации traffic events: {scenario_id}")


def execute_scenario(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    events_path: Path,
    traffic_path: Path,
    mock: bool,
) -> dict[str, Any]:
    append_event(events_path, build_execution_event(manifest, scenario, "scenario_started", "ok", {"mock": mock}))
    try:
        traffic_events = generate_traffic_events(manifest, scenario, mock)
        for event in traffic_events:
            append_event(traffic_path, event)
        details = {
            "requests_sent": len(traffic_events),
            "errors": sum(1 for event in traffic_events if event.get("status") == "error"),
            "mock": mock,
            "traffic_events": len(traffic_events),
        }
    except Exception as error:
        details = {"errors": 1, "message": str(error), "mock": mock, "traffic_events": 0}
        append_event(events_path, build_execution_event(manifest, scenario, "scenario_finished", "failed", details))
        return {"status": "failed", "details": details}

    append_event(events_path, build_execution_event(manifest, scenario, "scenario_finished", "completed", details))
    return {"status": "completed", "details": details}
