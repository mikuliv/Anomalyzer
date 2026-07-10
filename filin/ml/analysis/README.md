# Анализ Docker-прогонов Филин v0.2.1

Инструменты этой папки проверяют происхождение Docker datasets, рассчитывают смещение числовых признаков и формируют сводный отчёт по нескольким laboratory runs.

`execution_mode`, `synthetic` и `observation_source` являются metadata: они не входят в model features и используются только для контроля происхождения данных.

Docker client observations фиксируют результаты действий между контейнерами со стороны `traffic-client`. Они не являются независимыми сетевыми flow; следующий этап предусматривает подключение Zeek/Suricata.

## Аудит v0.2.2

Окна Docker datasets строятся по фактическим интервалам `scenario execution`, определённым парой `run_sequence + scenario_id`. Служебные события `scenario_executor` не являются client observations. После исправления временной разметки все Docker client observations корректно назначаются окнам соответствующих scenario executions. Однако каждый attack-класс пока представлен только одним независимым выполнением на run, поэтому данных недостаточно для достоверного обучения многоклассовой модели.

```powershell
python filin/ml/analysis/run_provenance.py --run-dir filin/lab/output/runs/run_docker_001 --windows-dataset filin/lab/output/datasets/windows_v0_1_run_docker_001.csv

python filin/ml/analysis/feature_drift.py --reference filin/lab/output/datasets/windows_v0_1_run_docker_001.csv --comparison filin/lab/output/datasets/windows_v0_1_run_docker_002.csv --target label --by-class --report filin/ml/reports/drift_docker_001_to_002.md --json-report filin/ml/reports/drift_docker_001_to_002.json
```
# Аудиты кампании v0.2.3

Аудиты проверяют происхождение, независимость executions, разделение train/test и готовность к ML.
