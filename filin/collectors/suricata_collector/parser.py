from __future__ import annotations

from typing import Any


def parse_eve_record(record: dict[str, Any]) -> dict[str, Any]:
    """Преобразует запись Suricata EVE JSON в единый словарь нормализованного события."""
    flow = record.get("flow") or {}
    alert = record.get("alert") or {}
    http = record.get("http") or {}
    dns = record.get("dns") or {}

    return {
        "timestamp": record.get("timestamp"),
        "event_source": "suricata",
        "log_type": record.get("event_type"),
        "source_ip": record.get("src_ip"),
        "source_port": record.get("src_port"),
        "destination_ip": record.get("dest_ip"),
        "destination_port": record.get("dest_port"),
        "protocol": record.get("proto"),
        "duration": flow.get("age"),
        "alert_signature": alert.get("signature"),
        "http_hostname": http.get("hostname"),
        "dns_query": dns.get("rrname"),
        "raw": record,
    }
