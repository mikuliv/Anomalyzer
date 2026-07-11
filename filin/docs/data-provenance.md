# Происхождение данных

## Источники

Client observations формируются traffic-client и используются только как отдельный контрольный источник. Sensor observations проходят цепочку:

```text
PCAP -> Zeek logs -> Zeek parser -> normalized sensor events -> marker-based correlation -> network_sensor_v0_3
```

События `network_sensor_v0_3` формируются только из фактически захваченного сетевого трафика. Traffic-client events используются для контрольного сравнения и не являются источником Zeek-событий или сетевых признаков.

## Markers и интервалы

Каждое execution получает реальные start/end HTTP markers внутри лабораторной сети. Они создают half-open sensor interval `[start, end)`. Execution markers используются только для корреляции и исключаются из model features. Корреляция не использует label, ожидаемые признаки или готовые client features.

## Контроль происхождения

Campaign roles разделяют train, test и robustness runs. Для PCAP, Zeek logs, normalized events и datasets сохраняются SHA-256. Provenance, split и duplicate audits проверяют отсутствие пересечений и leakage fields. Raw IP, hostname, URI, Zeek UID, marker metadata, label и execution identifiers не являются model features.
