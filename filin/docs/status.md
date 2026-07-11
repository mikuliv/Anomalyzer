# Текущий статус

| Компонент | Состояние | Последняя подтверждённая версия | Подтверждение | Ограничения |
| --- | --- | --- | --- | --- |
| Docker laboratory | Готово | v0.3 | Изолированные campaign runs | Только лабораторная сеть |
| Безопасные сценарии | Готово | v0.3 | Manifest и execution records | Не являются реальными атаками |
| Marker-aware executions | Готово | v0.3 | Start/end markers и sensor intervals | Markers не являются features |
| Passive capture | Готово | v0.3 | Capture-sidecar и PCAP audit | Нет production capture |
| PCAP storage | Готово | v0.3 | Docker-managed volume и hashes | Runtime artifacts вне Git |
| Zeek offline processing | Готово | v0.3 | PCAP→Zeek logs | Ограничено видимостью стенда |
| `network_sensor_v0_3` | Готово | v0.3 | Builders, validators, audits | Ограниченный набор сервисов |
| Independent train/test campaign | Готово | v0.3 | 6 train + 3 test runs | Лабораторный support |
| Frozen baseline evaluation | Экспериментально подтверждено | v0.3.1 | Pooled external test | Не production validation |
| Robustness evaluation | Экспериментально подтверждено | v0.3.2 | 12 external robustness-runs | Controlled shifts, identical metrics требуют осторожности |
| Backend model integration | Не начато | — | — | Нельзя считать backend готовым |
| MITRE ATT&CK mapping | Запланировано | — | — | Будущая работа |
| Sigma generation | Запланировано | — | — | Будущая работа |
| SIEM integration | Запланировано | — | — | Будущая работа |
| Production validation | Не выполнено | — | — | Отсутствуют production data и deployment procedure |
