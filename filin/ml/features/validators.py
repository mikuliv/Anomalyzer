from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from schema import get_feature_profile, get_forbidden_feature_columns, get_model_feature_columns


RELATION_TOLERANCE = 0.02


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"CSV-файл не найден: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def numeric_value(row: dict[str, str], column: str, row_number: int) -> float:
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"Пустое значение признака {column} в строке {row_number}.")
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"Признак {column} не приводится к числу в строке {row_number}.") from error
    if not math.isfinite(numeric):
        raise ValueError(f"Признак {column} имеет бесконечное значение в строке {row_number}.")
    return numeric


def assert_close(actual: float, expected: float, message: str) -> None:
    tolerance = max(RELATION_TOLERANCE, abs(expected) * RELATION_TOLERANCE)
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{message}: ожидается {expected:.4f}, получено {actual:.4f}")


def validate_windows_relations(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        duration = numeric_value(row, "duration_seconds", row_number)
        total_connections = numeric_value(row, "total_connections", row_number)
        total_packets = numeric_value(row, "total_packets", row_number)
        total_bytes = numeric_value(row, "total_bytes", row_number)
        bytes_in = numeric_value(row, "bytes_in", row_number)
        bytes_out = numeric_value(row, "bytes_out", row_number)
        packets_in = numeric_value(row, "packets_in", row_number)
        packets_out = numeric_value(row, "packets_out", row_number)

        assert_close(total_bytes, bytes_in + bytes_out, f"total_bytes не сходится в строке {row_number}")
        assert_close(total_packets, packets_in + packets_out, f"total_packets не сходится в строке {row_number}")

        if duration > 0:
            assert_close(
                numeric_value(row, "bytes_per_second", row_number),
                total_bytes / duration,
                f"bytes_per_second не сходится в строке {row_number}",
            )
            assert_close(
                numeric_value(row, "packets_per_second", row_number),
                total_packets / duration,
                f"packets_per_second не сходится в строке {row_number}",
            )
            assert_close(
                numeric_value(row, "syn_rate", row_number),
                numeric_value(row, "tcp_syn_count", row_number) / duration,
                f"syn_rate не сходится в строке {row_number}",
            )
            assert_close(
                numeric_value(row, "rst_rate", row_number),
                numeric_value(row, "tcp_rst_count", row_number) / duration,
                f"rst_rate не сходится в строке {row_number}",
            )

        if total_packets > 0:
            assert_close(
                numeric_value(row, "avg_packet_size", row_number),
                total_bytes / total_packets,
                f"avg_packet_size не сходится в строке {row_number}",
            )

        if total_connections > 0:
            assert_close(
                numeric_value(row, "short_connection_ratio", row_number),
                numeric_value(row, "short_connection_count", row_number) / total_connections,
                f"short_connection_ratio не сходится в строке {row_number}",
            )
            assert_close(
                numeric_value(row, "failed_connection_ratio", row_number),
                numeric_value(row, "failed_connection_count", row_number) / total_connections,
                f"failed_connection_ratio не сходится в строке {row_number}",
            )

        http_request_count = numeric_value(row, "http_request_count", row_number)
        if http_request_count > 0:
            assert_close(
                numeric_value(row, "http_4xx_rate", row_number),
                numeric_value(row, "http_4xx_count", row_number) / http_request_count,
                f"http_4xx_rate не сходится в строке {row_number}",
            )

        login_attempt_count = numeric_value(row, "login_attempt_count", row_number)
        if login_attempt_count > 0:
            assert_close(
                numeric_value(row, "failed_login_rate", row_number),
                numeric_value(row, "failed_login_count", row_number) / login_attempt_count,
                f"failed_login_rate не сходится в строке {row_number}",
            )


def validate_flows_relations(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        duration = numeric_value(row, "duration_seconds", row_number)
        if duration > 0:
            assert_close(
                numeric_value(row, "event_rate", row_number),
                numeric_value(row, "total_events", row_number) / duration,
                f"event_rate не сходится в строке {row_number}",
            )


def validate_dataset(path: Path, kind: str = "generic", feature_profile: str | None = None) -> None:
    columns, rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV-файл пустой: {path}")
    if "label" not in columns:
        raise ValueError("В CSV отсутствует обязательная колонка label.")
    if feature_profile and feature_profile.startswith("client_"):
        required = {"run_id","run_sequence","scenario_id","scenario_execution_key","window_index","window_start","window_end","window_duration_seconds","label","execution_mode","synthetic","observation_source","feature_profile","window_event_count","window_has_events"}
        missing = required-set(columns)
        if missing: raise ValueError("Отсутствует metadata: "+", ".join(sorted(missing)))
        profile_features=get_feature_profile(feature_profile)
        if set(profile_features)-set(columns): raise ValueError("Отсутствуют признаки профиля")
        for row in rows:
            if row["feature_profile"]!=feature_profile or row["execution_mode"]!="docker" or row["synthetic"].lower()!="false" or row["observation_source"]!="client": raise ValueError("Некорректная metadata client dataset")
            if float(row["window_event_count"])<=0 or float(row["window_duration_seconds"])<=0: raise ValueError("Некорректное окно")
        for row in rows:
            for key in ("http_4xx_rate","auth_failure_rate","http_event_ratio","tcp_event_ratio","dns_event_ratio","error_action_ratio","successful_action_ratio"):
                if key in row and not 0<=float(row[key])<=1: raise ValueError("Доля вне диапазона")
        return

    labels = {row.get("label") for row in rows}
    has_benign = "benign" in labels
    has_attack = any(label not in {None, "", "benign", "unknown"} for label in labels)
    if not has_benign or not has_attack:
        raise ValueError("В датасете должен быть хотя бы один benign и один attack label.")

    model_features = get_model_feature_columns(columns)
    forbidden = sorted(set(model_features) & set(get_forbidden_feature_columns()))
    if forbidden:
        raise ValueError(f"Запрещенные leakage-поля попали в признаки модели: {', '.join(forbidden)}")

    for row_number, row in enumerate(rows, start=2):
        for feature in model_features:
            numeric_value(row, feature, row_number)

    if kind == "windows":
        validate_windows_relations(rows)
    elif kind == "flows":
        validate_flows_relations(rows)


def print_validation_result(path: Path, kind: str = "generic", feature_profile: str | None = None) -> None:
    validate_dataset(path, kind=kind, feature_profile=feature_profile)
    print(f"Проверка датасета пройдена: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка CSV-датасетов признаков Филин.")
    parser.add_argument("--csv", required=True, help="Путь к CSV-файлу.")
    parser.add_argument(
        "--kind",
        choices=("generic", "windows", "flows"),
        default="generic",
        help="Тип датасета для дополнительных проверок.",
    )
    parser.add_argument("--feature-profile", default=None)
    args = parser.parse_args()
    print_validation_result(Path(args.csv), kind=args.kind, feature_profile=args.feature_profile)


if __name__ == "__main__":
    main()
