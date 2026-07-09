from __future__ import annotations

from typing import Any


def parse_zeek_record(record: dict[str, Any], log_type: str) -> dict[str, Any]:
    """Преобразует запись Zeek в единый словарь нормализованного события."""
    return {
        "timestamp": record.get("ts"),
        "event_source": "zeek",
        "log_type": log_type,
        "source_ip": record.get("id.orig_h"),
        "source_port": record.get("id.orig_p"),
        "destination_ip": record.get("id.resp_h"),
        "destination_port": record.get("id.resp_p"),
        "protocol": record.get("proto"),
        "service": record.get("service"),
        "duration": record.get("duration"),
        "bytes_in": record.get("resp_bytes"),
        "bytes_out": record.get("orig_bytes"),
        "raw": record,
    }


def parse_conn(record: dict[str, Any]) -> dict[str, Any]:
    return parse_zeek_record(record, "conn")


def parse_dns(record: dict[str, Any]) -> dict[str, Any]:
    event = parse_zeek_record(record, "dns")
    event["query"] = record.get("query")
    return event


def parse_http(record: dict[str, Any]) -> dict[str, Any]:
    event = parse_zeek_record(record, "http")
    event["method"] = record.get("method")
    event["uri"] = record.get("uri")
    event["status_code"] = record.get("status_code")
    return event


def parse_ssh(record: dict[str, Any]) -> dict[str, Any]:
    event = parse_zeek_record(record, "ssh")
    event["auth_success"] = record.get("auth_success")
    return event
