# Глоссарий

| Термин | Значение |
| --- | --- |
| execution | Одно фактическое выполнение лабораторного сценария. |
| run | Набор 13 executions, выполненный с одним seed и ролью campaign. |
| campaign | Набор связанных независимых runs. |
| train/test/robustness run | Run для выбора модели, внешней оценки или проверки frozen модели. |
| marker | Реальный start/end HTTP-запрос для временной привязки sensor observations. |
| sensor interval | Half-open интервал между парой markers. |
| client observation | Наблюдение traffic-client, не являющееся Zeek observation. |
| sensor observation | Событие, созданное из PCAP и Zeek logs. |
| feature profile | Зафиксированный список model features и metadata rules. |
| window | Агрегация observations в интервале одного execution. |
| PCAP | Фактически захваченный пакетный трафик. |
| normalized event | Событие в едином формате до aggregation. |
| assigned/background/excluded/ambiguous/unassigned | Результаты marker-aware корреляции событий. |
| provenance | Проверяемая цепочка происхождения артефакта и его hash. |
| split audit | Проверка отсутствия пересечений между ролями datasets. |
| frozen model | Модель, которую в robustness evaluation не обучают заново. |
| pooled evaluation | Метрики по объединённым строкам набора. |
| macro average | Среднее метрики по классам с равным весом классов. |
| attack macro recall | Macro recall только по attack-классам. |
| balanced accuracy | Средний recall по классам. |
| robustness shift | Контролируемое изменение topology, background, temporal или combined. |
