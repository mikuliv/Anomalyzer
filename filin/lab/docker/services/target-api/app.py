from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Филин target-api")


class LoginRequest(BaseModel):
    username: str
    password: str


def log_event(event_type: str, details: dict[str, Any]) -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "service": "target-api",
        "event_type": event_type,
        "details": details,
    }
    print(json.dumps(event, ensure_ascii=False), flush=True)


@app.get("/")
def root() -> dict[str, str]:
    log_event("root", {"status": "ok"})
    return {
        "service": "target-api",
        "status": "ok",
        "description": "Лабораторный API-сервис стенда Филин",
    }


@app.get("/health")
def health() -> dict[str, str]:
    log_event("health", {"status": "ok"})
    return {"status": "ok", "service": "target-api"}


@app.get("/api/items")
def items() -> dict[str, list[dict[str, str]]]:
    log_event("items_requested", {"count": 2})
    return {
        "items": [
            {"id": "item-1", "title": "Учебный объект"},
            {"id": "item-2", "title": "Лабораторная запись"},
        ]
    }


@app.get("/api/status")
def status() -> dict[str, str]:
    log_event("status_requested", {"status": "ready"})
    return {"status": "ready"}


@app.get("/api/profile/test-user")
def test_profile() -> dict[str, str]:
    log_event("profile_requested", {"username": "test-user"})
    return {"username": "test-user", "role": "lab-user"}


@app.post("/api/login")
def login(request: LoginRequest) -> dict[str, str]:
    valid = request.username == "test-user" and request.password == "test-password"
    log_event("login_attempt", {"username": request.username, "success": valid})
    if not valid:
        raise HTTPException(status_code=401, detail="Неверные тестовые учетные данные")
    return {"status": "ok", "username": "test-user"}
