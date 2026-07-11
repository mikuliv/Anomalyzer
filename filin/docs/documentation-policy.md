# Политика документации

Документация ведётся на русском языке. Канонические названия: проект «Филин», сетевой сенсор Zeek, профиль `network_sensor_v0_3`, лабораторный стенд, независимый run, sensor window, зафиксированная модель и проверка устойчивости.

Источники истины упорядочены так: код и schema, manifests/configuration, validators/audits, unit tests, runtime reports, Git history, затем документация. Метрики в технических таблицах используют точку и три знака после запятой.

Статусы: «Готово», «Частично готово», «Экспериментально подтверждено», «Запланировано», «Не начато». Реализованное и planned описываются отдельно. Нельзя заявлять production readiness, реальное-time detection, SIEM integration или завершённые MITRE/Sigma функции без подтверждения.

Runtime artifacts не коммитятся; документы ссылаются на их пути и checksums, но не копируют runtime JSON. При изменении experiment status обновляются `status.md`, `roadmap.md`, `experiments.md` и затронутый subsystem README.
