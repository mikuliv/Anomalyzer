# Сторонние зависимости и лицензии

> Таблица предназначена для технического inventory. Версии container images с `latest`, транзитивные OS packages и фактически скачанные image manifests требуют отдельного SBOM перед распространением. Совместимость ниже не является юридическим заключением.

## Принцип использования

Большинство библиотек подключаются как зависимости runtime или build environment; их исходный код не копируется в Git Filin. При распространении контейнеров, wheels, образов или vendor-кода обязательства могут отличаться от простого локального использования.

| Компонент | Версия в проекте | Использование | Лицензия / SPDX | Notices | Закрытая / Apache-2.0 / MPL-2.0 / GPL-3.0 |
| --- | --- | --- | --- | --- | --- |
| Python base image | `3.11-slim` | Backend и laboratory services | PSF-2.0 для CPython; Debian packages требуют отдельного inventory | License/notice образа | Обычно совместимо при соблюдении notices; OS packages проверять отдельно |
| FastAPI | `>=0.111,<1.0` | HTTP services/backend | MIT | Сохранить copyright/license при redistributing | Да / Да / Да / Да |
| Uvicorn | `>=0.30,<1.0` | ASGI server | BSD-3-Clause | Сохранить BSD notice | Да / Да / Да / Да |
| Pydantic | `>=2.7,<3.0` | Schemas/configuration | MIT | Сохранить notice | Да / Да / Да / Да |
| PyYAML | `>=6.0,<7.0` | Campaign/scenario YAML | MIT | Сохранить notice | Да / Да / Да / Да |
| requests | `>=2.32,<3.0` | traffic-client | Apache-2.0 | LICENSE/NOTICE при distribution | Да / Да / Да / Да |
| pandas | Не зафиксирована | Feature/data processing | BSD-3-Clause | Сохранить BSD notice | Да / Да / Да / Да |
| scikit-learn | Не зафиксирована | ML pipelines | BSD-3-Clause | Сохранить BSD notice | Да / Да / Да / Да |
| joblib | Не зафиксирована | Model serialization | BSD-3-Clause | Сохранить BSD notice | Да / Да / Да / Да |
| httpx | `<0.24` в historical requirements; не объявлен Filin requirements | Исторический scope | BSD-3-Clause | Если переносится, inventory обязателен | Условно; не является Filin dependency по текущим declarations |
| NumPy | Historical requirements | Исторический ML scope | BSD-3-Clause | При переносе — notice | Условно; не является declared Filin dependency |
| ONNX / ONNX Runtime | Config/предполагаемый runtime | Model interchange/inference prototype | MIT | При включении packages — notices | Да / Да / Да / Да |
| Zeek | `zeek/zeek:latest` | Offline PCAP processing и sensor capture base | BSD-3-Clause | Проверить конкретный image notice/digest | Обычно да; фиксировать digest |
| tcpdump / libpcap | Устанавливается в sensor image | Passive capture | BSD-3-Clause family; проверить package metadata конкретного образа | Package notices/SBOM | Обычно да; отдельные OS packages проверить |
| Nginx | `1.27-alpine` | target-web laboratory service | BSD-2-Clause | Сохранить notice при image distribution | Обычно да; Alpine packages проверить |
| Suricata | `jasonish/suricata:latest` | Optional sensor container | GPL-2.0-only (проверить конкретный image) | Source/notice obligations при redistributing image | Не считать безусловно совместимым с proprietary distribution; отдельный container boundary review |
| Elasticsearch | `8.15.0` | Optional laboratory stack | ELv2 для default distribution | Не скрывать notices; проверить image terms | Условно: ELv2 ограничивает managed service; не эквивалент Apache/MPL/GPL |
| Kibana | `8.15.0` | Optional laboratory UI | ELv2 для default distribution | Не скрывать notices; проверить image terms | Условно: ELv2 ограничивает managed service |
| Filebeat | `8.15.0` | Optional log shipping | Проверить конкретный distribution/image; вероятно ELv2/Apache components | Проверить manifest и notices | Условно до SBOM и image review |
| Docker Engine | External runtime | Containers/networks/volumes | Apache-2.0 for Engine | Docker notices | Да / Да / Да / Да |
| Docker Desktop | External workstation product | Windows local runtime | Docker Subscription Service Agreement | Проверить commercial entitlement | Не является library; licensing зависит от организации |

## Внешние образы и воспроизводимость

`zeek/zeek:latest` и `jasonish/suricata:latest` не закреплены digest. Это не доказывает нарушение, но создаёт риск воспроизводимости и усложняет license/SBOM audit: один и тот же tag может указывать на разные manifests. Перед external distribution закрепить image digest, собрать SBOM и сохранить license notices для каждого образа и его базовых OS packages.

Compose также объявляет `elastic/filebeat:8.15.0`, `elasticsearch:8.15.0` и `kibana:8.15.0`. Эти сервисы не образуют доказанной кодовой связи с historical Anomalyzer, но их лицензирование нельзя описывать как Apache-2.0 без проверки конкретной distribution. Elastic указывает ELv2 для default Elasticsearch/Kibana distribution и ограничения на предоставление продукта как managed service. См. [Elastic FAQ](https://www.elastic.co/licensing/elastic-license/faq/).

## Исторические зависимости

`Anomalyzer-main/model-training/requirements.txt` дополнительно объявляет imbalanced-learn, matplotlib, seaborn, torch и ONNX packages. Они не являются автоматически зависимостями Filin только потому, что находятся в соседнем историческом каталоге. Их нельзя переносить без отдельного inventory и review.

## Практические действия перед лицензированием/распространением

1. Зафиксировать Python versions через lock file и Docker image digests.
2. Сгенерировать SBOM для Filin images и runtime distribution.
3. Сохранить third-party notices, licenses и source-offer obligations там, где они применимы.
4. Решить, будет ли распространяться Compose stack с Suricata/Elastic images или они останутся development-only references.
5. Проверить условия Docker Desktop для организации: [официальные условия](https://docs.docker.com/subscription/desktop-license/).
