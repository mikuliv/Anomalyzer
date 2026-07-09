from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI


app = FastAPI(title="Филин control-api")


def log_event(event_type: str, details: dict[str, Any]) -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "service": "control-api",
        "event_type": event_type,
        "details": details,
    }
    print(json.dumps(event, ensure_ascii=False), flush=True)


@app.get("/health")
def health() -> dict[str, str]:
    log_event("health", {"status": "ok"})
    return {"status": "ok", "service": "control-api"}


@app.get("/beacon")
def get_beacon() -> dict[str, str]:
    log_event("beacon_get", {"purpose": "учебный heartbeat"})
    return {"status": "ok", "mode": "учебный heartbeat"}


@app.post("/beacon")
def post_beacon(payload: dict[str, Any] | None = None) -> dict[str, str]:
    log_event("beacon_post", {"payload": payload or {}, "purpose": "учебный heartbeat"})
    return {"status": "accepted"}
