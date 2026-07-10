from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from schema import get_feature_profile, get_model_feature_columns
from validators import validate_dataset


WINDOW_METADATA_COLUMNS = ["run_id", "run_sequence", "scenario_id", "scenario_execution_key", "window_index", "window_start", "window_end", "window_duration_seconds", "is_partial_window", "label", "label_type", "execution_mode", "synthetic", "observation_source", "feature_profile", "interval_source", "planned_scenario_duration_seconds", "actual_scenario_duration_seconds", "window_event_count", "window_has_events"]
FEATURE_COLUMNS = [
    "duration_seconds",
    "protocol_id",
    "dst_port",
    "total_connections",
    "total_packets",
    "total_bytes",
    "bytes_in",
    "bytes_out",
    "packets_in",
    "packets_out",
    "bytes_per_second",
    "packets_per_second",
    "avg_packet_size",
    "unique_dst_ports",
    "unique_dst_ips",
    "short_connection_count",
    "short_connection_ratio",
    "failed_connection_count",
    "failed_connection_ratio",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_rst_count",
    "tcp_fin_count",
    "syn_rate",
    "rst_rate",
    "http_request_count",
    "http_get_count",
    "http_post_count",
    "http_2xx_count",
    "http_4xx_count",
    "http_5xx_count",
    "http_4xx_rate",
    "unique_url_count",
    "suspicious_path_count",
    "file_download_count",
    "dns_query_count",
    "unique_dns_names",
    "dns_nxdomain_count",
    "login_attempt_count",
    "failed_login_count",
    "failed_login_rate",
    "mean_interarrival_time",
    "std_interarrival_time",
    "periodicity_score",
    "burst_score",
]
SUSPICIOUS_PATHS = {"/admin-test", "/debug-test", "/backup-test", "/old-login-test", "/not-found-test"}
DOWNLOAD_PATHS = {"/files/sample-small.txt", "/files/sample-config.json"}
PROTOCOL_IDS = {None: 0, "tcp": 1, "udp": 2, "http": 3, "dns": 4, "ssh": 5}


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Ошибка JSON в строке {line_number}: {error}") from error
    return events


def read_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"scenarios": []}


def scenario_bounds(scenario: dict[str, Any]) -> tuple[datetime, datetime]:
    start = parse_time(scenario.get("actual_started_at") or scenario["planned_started_at"])
    finish = parse_time(scenario.get("actual_finished_at") or scenario["planned_finished_at"])
    return start, finish


def choose_label(window_start: datetime, window_end: datetime, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps: list[dict[str, Any]] = []
    for scenario in scenarios:
        start, finish = scenario_bounds(scenario)
        if window_start < finish and window_end > start:
            overlaps.append(scenario)
    if not overlaps:
        return {
            "run_sequence": "",
            "scenario_id": "",
            "label": "unknown",
            "label_type": "unknown",
        }
    attack = [scenario for scenario in overlaps if scenario.get("type") == "attack"]
    selected = attack[0] if attack else overlaps[0]
    return {
        "run_sequence": selected.get("run_sequence", ""),
        "scenario_id": selected.get("scenario_id", ""),
        "label": selected.get("label", "unknown") if selected.get("type") == "attack" else "benign",
        "label_type": selected.get("type", "unknown"),
    }


def event_time(event: dict[str, Any]) -> datetime | None:
    value = event.get("timestamp")
    if not value:
        return None
    return parse_time(str(value))


def numeric(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stddev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def aggregate_window(events: list[dict[str, Any]], duration_seconds: int) -> dict[str, float]:
    http_events = [event for event in events if event.get("event_type") in {"http_request", "heartbeat_request"}]
    tcp_events = [event for event in events if event.get("event_type") == "tcp_connect_check"]
    dns_events = [event for event in events if event.get("event_type") == "dns_query"]
    auth_events = [event for event in events if event.get("event_type") == "auth_attempt"]
    admin_events = [event for event in events if event.get("event_type") == "admin_session_event"]
    bytes_in = sum(numeric(event.get("bytes_in")) for event in events)
    bytes_out = sum(numeric(event.get("bytes_out")) for event in events)
    total_packets = len(events)
    failed = [
        event
        for event in events
        if event.get("error") or event.get("status") in {"error", "closed"} or numeric(event.get("status_code")) >= 400
    ]
    short_connections = [event for event in events if numeric(event.get("latency_ms")) < 25]
    times = sorted(event_time(event) for event in events if event_time(event) is not None)
    gaps = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    std_gap = stddev(gaps, mean_gap)
    protocol_counts = Counter(event.get("protocol") for event in events)
    dominant_protocol = protocol_counts.most_common(1)[0][0] if protocol_counts else None
    ports = {event.get("target_port") for event in events if event.get("target_port") is not None}
    paths = {event.get("path") for event in http_events if event.get("path")}
    dns_names = {(event.get("details") or {}).get("query") or event.get("target_host") for event in dns_events}
    status_codes = [int(numeric(event.get("status_code"))) for event in http_events + auth_events if event.get("status_code")]
    http_4xx = sum(1 for code in status_codes if 400 <= code < 500)
    login_attempts = len(auth_events)
    failed_login = sum(1 for event in auth_events if event.get("auth_success") is False or numeric(event.get("status_code")) == 401)
    suspicious_paths = sum(1 for event in http_events if event.get("path") in SUSPICIOUS_PATHS)
    file_downloads = sum(1 for event in http_events if event.get("path") in DOWNLOAD_PATHS)
    closed_tcp = sum(1 for event in tcp_events if event.get("status") == "closed")

    result = {
        "duration_seconds": float(duration_seconds),
        "protocol_id": float(PROTOCOL_IDS.get(dominant_protocol, 0)),
        "dst_port": float(min(ports) if ports else 0),
        "total_connections": float(len(http_events) + len(tcp_events) + len(auth_events) + len(admin_events)),
        "total_packets": float(total_packets),
        "total_bytes": float(bytes_in + bytes_out),
        "bytes_in": float(bytes_in),
        "bytes_out": float(bytes_out),
        "packets_in": float(total_packets),
        "packets_out": float(total_packets),
        "bytes_per_second": float((bytes_in + bytes_out) / duration_seconds) if duration_seconds else 0.0,
        "packets_per_second": float(total_packets / duration_seconds) if duration_seconds else 0.0,
        "avg_packet_size": float((bytes_in + bytes_out) / total_packets) if total_packets else 0.0,
        "unique_dst_ports": float(len(ports)),
        "unique_dst_ips": float(len({event.get("target_host") for event in events if event.get("target_host")})),
        "short_connection_count": float(len(short_connections)),
        "short_connection_ratio": float(len(short_connections) / total_packets) if total_packets else 0.0,
        "failed_connection_count": float(len(failed)),
        "failed_connection_ratio": float(len(failed) / total_packets) if total_packets else 0.0,
        "tcp_syn_count": float(len(tcp_events)),
        "tcp_ack_count": float(max(0, len(tcp_events) - closed_tcp)),
        "tcp_rst_count": float(closed_tcp),
        "tcp_fin_count": float(max(0, len(tcp_events) - closed_tcp)),
        "syn_rate": float(len(tcp_events) / duration_seconds) if duration_seconds else 0.0,
        "rst_rate": float(closed_tcp / duration_seconds) if duration_seconds else 0.0,
        "http_request_count": float(len(http_events)),
        "http_get_count": float(sum(1 for event in http_events if event.get("method") == "GET")),
        "http_post_count": float(sum(1 for event in http_events if event.get("method") == "POST")),
        "http_2xx_count": float(sum(1 for code in status_codes if 200 <= code < 300)),
        "http_4xx_count": float(http_4xx),
        "http_5xx_count": float(sum(1 for code in status_codes if 500 <= code < 600)),
        "http_4xx_rate": float(http_4xx / len(status_codes)) if status_codes else 0.0,
        "unique_url_count": float(len(paths)),
        "suspicious_path_count": float(suspicious_paths),
        "file_download_count": float(file_downloads),
        "dns_query_count": float(len(dns_events)),
        "unique_dns_names": float(len(dns_names)),
        "dns_nxdomain_count": float(sum(1 for event in dns_events if event.get("status") == "nxdomain")),
        "login_attempt_count": float(login_attempts),
        "failed_login_count": float(failed_login),
        "failed_login_rate": float(failed_login / login_attempts) if login_attempts else 0.0,
        "mean_interarrival_time": float(mean_gap),
        "std_interarrival_time": float(std_gap),
        "periodicity_score": float(1 / (1 + std_gap)) if gaps else 0.0,
        "burst_score": float(total_packets / max(1, len({format_time(time)[:16] for time in times}))) if times else 0.0,
    }
    return result


def aggregate_client_window(events: list[dict[str, Any]], duration_seconds: int) -> dict[str, float]:
    base = aggregate_window(events, duration_seconds)
    http = [event for event in events if event.get("event_type") in {"http_request", "auth_attempt", "heartbeat_request"}]
    tcp = [event for event in events if event.get("event_type") == "tcp_connect_check"]
    dns = [event for event in events if event.get("event_type") == "dns_resolution"]
    admin = [event for event in events if event.get("event_type") == "admin_tcp_session_check"]
    auth = [event for event in events if event.get("event_type") == "auth_attempt"]
    errors = [event for event in events if event.get("status") in {"error", "closed", "timeout"} or event.get("error")]
    latencies = [numeric(event.get("latency_ms")) for event in events]
    sizes = [numeric(event.get("bytes_in")) for event in events]
    count = len(events)
    return {
        "event_count": float(count), "successful_action_count": float(sum(event.get("status") in {"ok", "open"} for event in events)), "error_action_count": float(len(errors)), "timeout_count": float(sum(event.get("status") == "timeout" for event in events)),
        "http_request_count": float(len(http)), "http_get_count": float(sum(event.get("method") == "GET" for event in http)), "http_post_count": float(sum(event.get("method") == "POST" for event in http)), "http_2xx_count": float(sum(200 <= numeric(event.get("status_code")) < 300 for event in http)), "http_4xx_count": float(sum(400 <= numeric(event.get("status_code")) < 500 for event in http)), "http_5xx_count": float(sum(500 <= numeric(event.get("status_code")) < 600 for event in http)), "http_4xx_rate": float(sum(400 <= numeric(event.get("status_code")) < 500 for event in http) / len(http)) if http else 0.0,
        "auth_attempt_count": float(len(auth)), "auth_failure_count": float(sum(event.get("auth_success") is False for event in auth)), "auth_failure_rate": float(sum(event.get("auth_success") is False for event in auth) / len(auth)) if auth else 0.0,
        "tcp_check_count": float(len(tcp)), "tcp_open_count": float(sum(event.get("status") == "open" for event in tcp)), "tcp_closed_count": float(sum(event.get("status") == "closed" for event in tcp)), "tcp_timeout_count": float(sum(event.get("status") == "timeout" for event in tcp)), "admin_tcp_session_count": float(len(admin)),
        "dns_resolution_count": float(len(dns)), "dns_success_count": float(sum(event.get("status") == "ok" for event in dns)), "dns_error_count": float(sum(event.get("status") == "error" for event in dns)), "unique_dns_name_count": float(len({event.get("target_host") for event in dns})), "unique_resolved_address_count": float(len({address for event in dns for address in (event.get("details") or {}).get("resolved_addresses", [])})), "heartbeat_count": float(sum(event.get("event_type") == "heartbeat_request" for event in events)),
        "bytes_received": float(sum(sizes)), "bytes_sent": float(sum(numeric(event.get("bytes_out")) for event in events)), "mean_response_size": float(sum(sizes) / count) if count else 0.0, "max_response_size": float(max(sizes, default=0)), "latency_mean": float(sum(latencies) / count) if count else 0.0, "latency_median": float(sorted(latencies)[count // 2]) if count else 0.0, "latency_std": float(stddev(latencies, sum(latencies) / count)) if count else 0.0, "latency_p95": float(sorted(latencies)[max(0, math.ceil(count * .95) - 1)]) if count else 0.0, "latency_max": float(max(latencies, default=0)),
        "mean_interarrival_time": base["mean_interarrival_time"], "std_interarrival_time": base["std_interarrival_time"], "periodicity_score": base["periodicity_score"], "burst_score": base["burst_score"], "http_event_ratio": float(len(http) / count) if count else 0.0, "tcp_event_ratio": float(len(tcp) / count) if count else 0.0, "dns_event_ratio": float(len(dns) / count) if count else 0.0,
        "error_action_ratio": float(len(errors) / count) if count else 0.0, "successful_action_ratio": float(sum(event.get("status") in {"ok", "open"} for event in events) / count) if count else 0.0,
        "suspicious_path_count": base["suspicious_path_count"], "file_download_count": base["file_download_count"], "unique_url_count": base["unique_url_count"], "unique_target_host_count": float(len({event.get("target_host") for event in events if event.get("target_host")})), "unique_target_port_count": float(len({event.get("target_port") for event in events if event.get("target_port") is not None})),
    }


def build_scenario_execution_windows(manifest: dict[str, Any], events: list[dict[str, Any]], window_seconds: int, feature_profile: str, empty_window_policy: str, tolerance: float, out_policy: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    assigned: set[str] = set()
    unassigned = 0
    for scenario in manifest.get("scenarios", []):
        sequence = scenario.get("run_sequence")
        scenario_events = [event for event in events if event.get("run_sequence") == sequence and event.get("scenario_id") == scenario.get("scenario_id")]
        start, finish = scenario_bounds(scenario)
        observed_times = [event_time(event) for event in scenario_events if event_time(event) is not None]
        if observed_times:
            start = min(start, min(observed_times))
            finish = max(finish, max(observed_times) + timedelta(seconds=tolerance))
        if finish <= start:
            raise ValueError(f"Фактическая длительность сценария должна быть положительной: {scenario.get('scenario_id')}")
        key = f"{manifest.get('run_id')}:{sequence}:{scenario.get('scenario_id')}"
        current, index = start, 0
        while current < finish:
            end = min(current + timedelta(seconds=window_seconds), finish)
            window_events = []
            for event in scenario_events:
                timestamp = event_time(event)
                if timestamp is None:
                    continue
                if current - timedelta(seconds=tolerance) <= timestamp <= end + timedelta(seconds=tolerance):
                    event_id = str(event.get("event_id") or f"{sequence}:{scenario.get('scenario_id')}:{timestamp.isoformat()}:{len(window_events)}")
                    if event_id in assigned:
                        continue
                    if timestamp < current or timestamp >= end:
                        if abs((timestamp - current).total_seconds()) > tolerance and abs((timestamp - end).total_seconds()) > tolerance:
                            if out_policy == "error":
                                raise ValueError(f"Событие вне интервала сценария: {event_id}")
                            if out_policy == "drop":
                                continue
                    assigned.add(event_id)
                    window_events.append(event)
            if not window_events and empty_window_policy == "error":
                raise ValueError(f"Пустое окно сценария: {key}")
            if window_events or empty_window_policy == "keep":
                duration = (end - current).total_seconds()
                row = {"run_id": manifest.get("run_id", ""), "run_sequence": sequence, "scenario_id": scenario.get("scenario_id", ""), "scenario_execution_key": key, "window_index": index, "window_start": format_time(current), "window_end": format_time(end), "window_duration_seconds": duration, "is_partial_window": duration < window_seconds, "label": scenario.get("label", "benign") if scenario.get("type") == "attack" else "benign", "label_type": scenario.get("type"), "execution_mode": manifest.get("execution_mode"), "synthetic": False, "observation_source": "client", "feature_profile": feature_profile, "interval_source": "manifest_actual", "planned_scenario_duration_seconds": scenario.get("duration_seconds"), "actual_scenario_duration_seconds": (finish-start).total_seconds(), "window_event_count": len(window_events), "window_has_events": bool(window_events)}
                row.update(aggregate_client_window(window_events, duration))
                rows.append(row)
            current, index = end, index + 1
    input_events = [event for event in events if event.get("event_source") != "scenario_executor"]
    unassigned = max(0, len(input_events) - len(assigned))
    return rows, {"total_input_events": len(input_events), "assigned_events": len(assigned), "unassigned_events": unassigned, "duplicated_assignments": 0}


def build_windows_dataset(manifest_path: Path, events_path: Path, output_path: Path, window_seconds: int, feature_profile: str = "legacy_v0_1", empty_window_policy: str = "keep", timestamp_tolerance_seconds: float = 1.0, out_of_interval_policy: str = "error") -> int:
    manifest = read_manifest(manifest_path)
    scenarios = manifest.get("scenarios", [])
    events = [event for event in read_jsonl(events_path) if event.get("event_source") != "scenario_executor"]
    timed_events = [(event_time(event), event) for event in events if event_time(event) is not None]
    if not scenarios:
        raise ValueError("Manifest не содержит сценариев.")
    if feature_profile.startswith("client_"):
        rows, assignment = build_scenario_execution_windows(manifest, events, window_seconds, feature_profile, empty_window_policy, timestamp_tolerance_seconds, out_of_interval_policy)
        if assignment["unassigned_events"] or assignment["duplicated_assignments"]:
            raise ValueError(f"Ошибка назначения событий: {assignment}")
        starts = []
    else:
        starts = [scenario_bounds(scenario)[0] for scenario in scenarios]
    finishes = [scenario_bounds(scenario)[1] for scenario in scenarios]
    if not feature_profile.startswith("client_"):
      start = min(starts)
      finish = max(finishes)
      rows = []
      current = start
      while current < finish:
        window_end = current + timedelta(seconds=window_seconds)
        window_events = [event for time_value, event in timed_events if time_value and current <= time_value < window_end]
        label_data = choose_label(current, window_end, scenarios)
        if not window_events and empty_window_policy == "drop":
            current = window_end
            continue
        if not window_events and empty_window_policy == "error":
            raise ValueError(f"Обнаружено пустое окно: {format_time(current)}")
        row: dict[str, Any] = {
            "run_id": manifest.get("run_id", ""),
            "run_sequence": label_data["run_sequence"],
            "scenario_id": label_data["scenario_id"],
            "window_start": format_time(current),
            "window_end": format_time(window_end),
            "label": label_data["label"],
            "label_type": label_data["label_type"],
            "execution_mode": manifest.get("execution_mode", "mock"),
            "synthetic": manifest.get("execution_mode", "mock") == "mock",
            "observation_source": "generator" if manifest.get("execution_mode", "mock") == "mock" else "client",
            "feature_profile": feature_profile,
            "window_event_count": len(window_events),
            "window_has_events": bool(window_events),
        }
        if feature_profile.startswith("client_"):
            row.update(aggregate_client_window(window_events, duration_seconds=window_seconds))
        else:
            row.update(aggregate_window(window_events, duration_seconds=window_seconds))
        rows.append(row)
        current = window_end

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        profile_features = get_feature_profile(feature_profile)
        fieldnames = WINDOW_METADATA_COLUMNS + (profile_features or FEATURE_COLUMNS)
        rows = [{field: row.get(field) for field in fieldnames} for row in rows]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_dataset(output_path)
    model_features = get_model_feature_columns(WINDOW_METADATA_COLUMNS + (get_feature_profile(feature_profile) or FEATURE_COLUMNS))
    print(f"Window-level датасет записан: {output_path}")
    print(f"Количество строк: {len(rows)}")
    print(f"Количество модельных признаков: {len(model_features)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение window-level датасета признаков Филин v0.1.")
    parser.add_argument("--run-dir", default=None, help="Папка одного laboratory run.")
    parser.add_argument("--manifest", default=None, help="Путь к scenario_manifest.yaml.")
    parser.add_argument("--events", default=None, help="Путь к normalized_events.jsonl.")
    parser.add_argument("--output", required=True, help="Путь к выходному CSV.")
    parser.add_argument("--window-seconds", type=int, default=60, help="Размер временного окна в секундах.")
    parser.add_argument("--feature-profile", default="legacy_v0_1", choices=("legacy_v0_1", "client_core_v0_2", "client_extended_v0_2"))
    parser.add_argument("--empty-window-policy", default="keep", choices=("keep", "drop", "error"))
    parser.add_argument("--mixed-label-policy", default="error", choices=("error", "dominant", "split"))
    parser.add_argument("--timestamp-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--out-of-interval-policy", default="error", choices=("error", "warn", "drop"))
    args = parser.parse_args()
    if args.window_seconds <= 0:
        raise ValueError("window-seconds должен быть положительным числом.")
    if args.run_dir:
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "scenario_manifest.yaml"
        events_path = run_dir / "normalized_events.jsonl"
    else:
        if not args.manifest or not args.events:
            raise ValueError("Нужно указать --run-dir или оба параметра --manifest и --events.")
        manifest_path = Path(args.manifest)
        events_path = Path(args.events)
    build_windows_dataset(manifest_path, events_path, Path(args.output), args.window_seconds, args.feature_profile, args.empty_window_policy, args.timestamp_tolerance_seconds, args.out_of_interval_policy)


if __name__ == "__main__":
    main()
