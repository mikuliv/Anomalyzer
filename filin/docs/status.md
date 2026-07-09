# Статус компонентов

| Компонент | Статус | Комментарий |
| --- | --- | --- |
| Исторический Anomalyzer | готово | Сохранен как базовая версия колледжной ВКР в `Anomalyzer-main/`. |
| Backend API | прототип | Есть FastAPI endpoint-ы для health, predict, incidents, Sigma и rules validation. |
| Incident processing | прототип | Формируется карточка инцидента с in-memory хранилищем. |
| MITRE mapping | прототип | Есть первичная таблица соответствий классов и техник. |
| Sigma draft generator | прототип | Генерируется Sigma-кандидат, требующий лабораторной проверки. |
| Лабораторный manifest | готово | Manifest v0.3 фиксирует план, разметку и статусы выполнения. |
| Natural schedule | готово | Attack-сценарии вплетаются в benign-фон. |
| Mock execute | готово | Формирует синтетические лабораторные события без сетевой активности. |
| Docker services | прототип | Есть target-web, target-api, control-api, traffic-client и инфраструктурные сервисы. |
| Traffic events | готово | Формируется `traffic_events.jsonl` с учебными событиями активности. |
| Normalized events | готово | `normalize_events.py` объединяет execution и traffic events. |
| Dataset report | готово | Отчет учитывает manifest, execution events, traffic events и normalized events. |
| Feature extraction | прототип | Есть feature catalog, schema, validators, windows и flows builders. |
| Model training | планируется | Требуется train/test split, baseline и эксперименты. |
| Model evaluation | планируется | Требуются precision, recall, F1, confusion matrix, ROC-AUC/PR-AUC. |
| ONNX export | частично | Есть заготовка, реальная модель пока не экспортируется. |
| Dashboard/SIEM export | планируется | Требуются web-ui, Kibana/SIEM сценарии и проверка Sigma-кандидатов. |
| Zeek collector | частично | Есть заготовка коллектора, требуется реальный разбор логов. |
| Suricata collector | частично | Есть заготовка коллектора, требуется реальный разбор EVE JSON. |
| VMware стенд | планируется | Нужен для более реалистичного сбора трафика и PCAP. |
