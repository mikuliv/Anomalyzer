# Runtime datasets

## Назначение

Описание воспроизводимо формируемых feature datasets проекта «Филин».

## Что реализовано

Поддерживаются `client_core_v0_2`, `client_extended_v0_2` и `network_sensor_v0_3`. Основной размер sensor window — 60 секунд; фактические короткие executions могут образовывать одну активную строку.

## Входные данные и выходные данные

Client datasets строятся из client observations. Sensor datasets строятся из correlated Zeek observations. CSV, PCAP, JSONL, indexes и runtime reports не коммитятся.

## Metadata и признаки

Metadata содержит provenance, role, execution/window и label fields. Model features исключают label, IDs, campaign/robustness metadata, marker fields, raw IP/hostname/URI/port identifier и Zeek UID.

## Labels и роли

Labels: `benign`, `port_scan`, `auth_failures`, `web_probe`, `low_rate_dos`, `beacon_simulation`. Train, test и robustness roles разделены audit-ами; hashes включаются в indexes.

## Проверки

Отсутствие runtime dataset в Git не означает отсутствие поддержки профиля. Datasets формируются воспроизводимым campaign pipeline и проверяются по индексам и SHA-256.

## Ограничения

Datasets описывают controlled laboratory observations и не являются production corpus.

## Связанные документы

[Происхождение данных](../docs/data-provenance.md), [profiles](../ml/features/README.md), [эксперименты](../docs/experiments.md).
