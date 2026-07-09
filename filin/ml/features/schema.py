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
    "label",
    "label_type",
]

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
