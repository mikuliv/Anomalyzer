from __future__ import annotations

from collections.abc import Iterable
from typing import Any


METADATA_COLUMNS = [
    "run_id",
    "run_sequence",
    "scenario_id",
    "window_start",
    "window_end",
    "source_role",
    "target_role",
    "event_type",
    "execution_mode",
    "synthetic",
    "observation_source",
    "label",
    "label_type",
    "feature_profile",
    "window_event_count",
    "window_has_events",
    "scenario_execution_key",
    "window_index",
    "window_duration_seconds",
    "is_partial_window",
    "interval_source",
    "planned_scenario_duration_seconds",
    "actual_scenario_duration_seconds",
    "campaign_id", "campaign_version", "campaign_role", "campaign_run_index", "campaign_seed",
    "execution_id", "scenario_variant_id", "scenario_parameter_hash",
]

CLIENT_CORE_V0_2 = [
    "event_count", "successful_action_count", "error_action_count", "timeout_count",
    "http_request_count", "http_get_count", "http_post_count", "http_2xx_count", "http_4xx_count", "http_5xx_count", "http_4xx_rate",
    "auth_attempt_count", "auth_failure_count", "auth_failure_rate", "tcp_check_count", "tcp_open_count", "tcp_closed_count", "tcp_timeout_count", "admin_tcp_session_count",
    "dns_resolution_count", "dns_success_count", "dns_error_count", "unique_dns_name_count", "unique_resolved_address_count", "heartbeat_count",
    "bytes_received", "bytes_sent", "mean_response_size", "max_response_size", "latency_mean", "latency_median", "latency_std", "latency_p95", "latency_max",
    "mean_interarrival_time", "std_interarrival_time", "periodicity_score", "burst_score", "http_event_ratio", "tcp_event_ratio", "dns_event_ratio", "error_action_ratio", "successful_action_ratio",
]
CLIENT_EXTENDED_V0_2 = CLIENT_CORE_V0_2 + ["suspicious_path_count", "file_download_count", "unique_url_count", "unique_target_host_count", "unique_target_port_count"]
PACKET_FEATURES = {"total_packets", "packets_in", "packets_out", "tcp_syn_count", "tcp_ack_count", "tcp_rst_count", "tcp_fin_count", "syn_rate", "rst_rate", "protocol_id", "dst_port"}
NETWORK_SENSOR_V0_3 = ["flow_count","tcp_flow_count","udp_flow_count","icmp_flow_count","unique_destination_ip_count","unique_destination_port_count","unique_service_count","flow_duration_mean","flow_duration_median","flow_duration_std","flow_duration_p95","flow_duration_max","orig_bytes_total","resp_bytes_total","total_bytes","orig_bytes_mean","resp_bytes_mean","orig_resp_bytes_ratio","orig_packets_total","resp_packets_total","total_packets","orig_packets_mean","resp_packets_mean","orig_resp_packets_ratio","successful_connection_count","failed_connection_count","rejected_connection_count","reset_connection_count","connection_success_rate","connection_failure_rate","unique_conn_state_count","http_request_count","http_get_count","http_post_count","http_2xx_count","http_4xx_count","http_5xx_count","http_error_rate","unique_http_host_count","unique_uri_count","http_request_body_bytes","http_response_body_bytes","dns_query_count","dns_success_count","dns_error_count","unique_dns_query_count","unique_dns_answer_count","dns_error_rate","flow_interarrival_mean","flow_interarrival_std","flow_periodicity_score","flow_burst_score"]


def get_feature_profile(name: str) -> list[str]:
    if name == "client_core_v0_2":
        return list(CLIENT_CORE_V0_2)
    if name == "client_extended_v0_2":
        return list(CLIENT_EXTENDED_V0_2)
    if name == "legacy_v0_1":
        return []
    if name == "network_sensor_v0_3":
        return list(NETWORK_SENSOR_V0_3)
    raise ValueError(f"Неизвестный профиль признаков: {name}")

FORBIDDEN_FEATURE_COLUMNS = [
    "scenario_id",
    "run_sequence",
    "planned_started_at",
    "planned_finished_at",
    "actual_started_at",
    "actual_finished_at",
    "label",
    "label_type",
    "mitre_technique_id",
    "execution_mode",
    "synthetic",
    "observation_source",
    "campaign_id", "campaign_version", "campaign_role", "campaign_run_index", "campaign_seed",
    "execution_id", "scenario_variant_id", "scenario_parameter_hash",
]


def get_metadata_columns() -> list[str]:
    return list(METADATA_COLUMNS)


def get_forbidden_feature_columns() -> list[str]:
    return list(FORBIDDEN_FEATURE_COLUMNS)


def get_model_feature_columns(rows_or_columns: Iterable[Any]) -> list[str]:
    columns = list(rows_or_columns)
    if columns and isinstance(columns[0], dict):
        columns = list(columns[0].keys())
    excluded = set(get_metadata_columns()) | set(get_forbidden_feature_columns())
    return [str(column) for column in columns if str(column) not in excluded]
