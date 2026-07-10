from __future__ import annotations

import hashlib
import json
import random
from typing import Any


CAMPAIGN_FIELDS = ("campaign_id", "campaign_version", "campaign_role", "campaign_run_index", "campaign_seed")
TEMPORAL_MINIMUMS = {"low_rate_dos": 6, "beacon_simulation": 8}


def stable_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_campaign(data: dict[str, Any]) -> None:
    runs = data.get("runs", [])
    if data.get("campaign_id") != "filin-v0.2.3-independent-executions":
        raise ValueError("Некорректный campaign_id.")
    if len(runs) != 9:
        raise ValueError("Кампания должна содержать ровно девять независимых runs.")
    ids = [item.get("run_id") for item in runs]
    seeds = [item.get("random_seed") for item in runs]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Run ID должны быть заполнены и уникальны.")
    if any(not isinstance(item, int) for item in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds должны быть целыми и уникальными.")
    roles = [item.get("role") for item in runs]
    if any(role not in {"train", "test"} for role in roles):
        raise ValueError("Разрешены только роли train и test.")
    if roles.count("train") != 6 or roles.count("test") != 3:
        raise ValueError("Нужно шесть train и три test runs.")


def campaign_metadata(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "campaign_id": campaign["campaign_id"],
        "campaign_version": campaign["campaign_version"],
        "campaign_role": run["role"],
        "campaign_run_index": run["run_index"],
        "campaign_seed": run["random_seed"],
    }


def scenario_parameters(seed: int, scenario_id: str) -> dict[str, Any]:
    """Возвращает воспроизводимый безопасный вариант одного сценария."""
    rng = random.Random(f"filin-v023:{seed}:{scenario_id}")
    label = scenario_id.removeprefix("attack_")
    if label == "port_scan":
        result = {"port_count": rng.randint(4, 12), "base_delay_ms": rng.randint(250, 700)}
    elif label == "auth_failures":
        result = {"attempt_count": rng.randint(5, 12), "base_delay_ms": rng.randint(300, 800)}
    elif label == "web_probe":
        result = {"path_count": rng.randint(4, 10), "base_delay_ms": rng.randint(250, 700)}
    elif label == "low_rate_dos":
        count = rng.randint(10, 20)
        result = {"request_count": count, "max_rate": round(rng.uniform(1.0, 2.5), 2), "base_delay_ms": rng.randint(400, 900), "preserve_actual_timing": True, "minimum_actual_duration_seconds": max(6, int(count / 2.5))}
    elif label == "beacon_simulation":
        count = rng.randint(8, 12)
        interval = rng.randint(1000, 1600)
        result = {"heartbeat_count": count, "base_interval_ms": interval, "preserve_actual_timing": True, "minimum_actual_duration_seconds": max(8, int((count - 1) * interval / 1000) + 1)}
    else:
        result = {"action_count": rng.randint(3, 8), "base_delay_ms": rng.randint(250, 700), "jitter": round(rng.uniform(0.05, 0.2), 3)}
    return result


def build_execution_metadata(campaign: dict[str, Any], run: dict[str, Any], sequence: int, scenario_id: str) -> dict[str, Any]:
    parameters = scenario_parameters(int(run["random_seed"]), scenario_id)
    parameter_hash = stable_hash({"scenario_id": scenario_id, "parameters": parameters})
    return {
        **campaign_metadata(campaign, run),
        "execution_id": f"{run['run_id']}:{sequence}:{scenario_id}",
        "scenario_variant_id": f"{scenario_id}:{parameter_hash[:12]}",
        "scenario_parameter_hash": parameter_hash,
        "scenario_parameters": parameters,
    }
