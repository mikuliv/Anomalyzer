# Воспроизводимость

## Требования

Нужны Docker Engine/Docker Desktop и Python с зависимостями проекта. PCAP создаётся и проверяется внутри Docker-managed storage; runtime artifacts находятся в `filin/lab/output/` и `filin/ml/reports/` и не коммитятся.

## Быстрые проверки

```powershell
python filin/tools/docs/validate_documentation.py --strict
python -m compileall filin/tools filin/ml/tests
python -m unittest discover -s filin/ml/tests -p "test_*.py"
```

## Проверка CLI

```powershell
python filin/lab/sensor/run_v0_3_sensor_stage.py --help
python filin/lab/robustness/run_v0_3_2_stage.py --help
python filin/ml/features/validators.py --help
```

Campaign runners поддерживают `--resume`; он предназначен для проверки и продолжения существующих attempts. Не запускайте исторические 9 sensor-runs или 12 robustness-runs повторно только ради документации.

## Контроль целостности

Capture и Zeek processing сопоставляются по SHA-256 PCAP. Dataset indexes, provenance и split audits проверяют итоговые артефакты. Команды тяжёлых кампаний документируются в README соответствующих подсистем, а не выполняются как часть быстрой проверки.
