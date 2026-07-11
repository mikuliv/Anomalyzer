# Эксперименты и метрики

Технические таблицы используют точку как десятичный разделитель и три знака после запятой.

| Версия | Вопрос | Train/test и grouping | Результат | Ограничение |
| --- | --- | --- | --- | --- |
| v0.2.4 | Переносятся ли client profiles между независимыми Docker-runs? | Независимые train/test runs; Leave-One-Train-Run-Out или фактический CV. | Client pipeline проверен технически. | Недостаточная независимая поддержка attack-классов на ранних данных. |
| v0.3.1 | Какой профиль лучше переносится на независимый test? | 6 train и 3 test sensor-runs; test не использовался для model selection. | `network_sensor_v0_3` рекомендован; feature fusion отсутствовала. | Лабораторная среда ограничена. |
| v0.3.2 | Сохраняется ли frozen baseline при controlled shifts? | 12 robustness-runs, 156 windows; только transform/predict/evaluation. | Policy пройдена. | Это не production validation. |

## v0.3.1

| Профиль | Train CV macro F1 | Pooled test macro F1 | Balanced accuracy | Attack macro recall | Support test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `client_core_v0_2` | 0.989 | 0.024 | 0.167 | 0.200 | 39 |
| `network_sensor_v0_3` | 0.749 | 0.918 | 0.972 | 1.000 | 39 |

`network_sensor_v0_3` превзошёл DummyClassifier по macro F1 (`0.918` против `0.127`). Модель выбиралась только по независимым train-runs. Test- и robustness-runs не участвовали в выборе признаков, preprocessing или настройке гиперпараметров.

## v0.3.2

Зафиксированная `LogisticRegression` с `SimpleImputer(strategy="median")` и `StandardScaler` оценена на 12 robustness-runs: по три topology, background, temporal и combined. Pooled macro F1 — `0.933`, balanced accuracy — `0.979`, attack macro recall — `1.000`; support — 156 windows. Повторное обучение не выполнялось.

Метрики всех 12 runs одинаковы в сохранённом runtime report. Это наблюдение требует осторожной интерпретации и не является подтверждением широкого обобщения.
