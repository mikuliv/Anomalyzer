# Архитектура

## Реализованная архитектура

```mermaid
flowchart LR
  S[Безопасный сценарий] --> T[traffic-client]
  T --> M[Start/end markers]
  T --> C[capture-sidecar]
  C --> P[PCAP: Docker named volume]
  P --> Z[Offline Zeek]
  Z --> L[Zeek logs]
  L --> N[Нормализация]
  M --> I[Sensor-aligned interval]
  N --> R[Корреляция]
  I --> R
  R --> F[Feature builder]
  F --> D[Dataset]
  D --> A[Audits и ML evaluation]
```

`traffic-client` выполняет безопасные действия в изолированной сети. Capture-sidecar наблюдает тот же network namespace; PCAP является первичным источником sensor observations. Zeek logs нормализуются до корреляции. Execution markers используются только для временной привязки и исключаются из feature aggregation.

## Campaign separation

```mermaid
flowchart TB
  C[Campaign] --> TR[Независимые train runs]
  C --> TE[Независимые test runs]
  TR --> SEL[Model selection]
  TE --> EXT[External evaluation]
  EXT --> FR[Зафиксированная модель]
  FR --> RB[Robustness runs: только predict/evaluation]
```

## Концептуальная будущая архитектура

```mermaid
flowchart LR
  D[Проверенный sensor dataset] --> M[Модель]
  M -. будущая работа .-> IC[Incident card]
  IC -. будущая работа .-> MA[MITRE ATT&CK mapping]
  MA -. будущая работа .-> SG[Sigma draft]
  SG -. будущая работа .-> SIEM[SIEM integration]
  IC -. будущая работа .-> UI[Analyst interface]
```

Концептуальная архитектура. На текущем этапе полностью не реализована.
