# ML-раздел Филин

Раздел содержит заготовки для обучения, оценки и экспорта моделей обнаружения сетевых аномалий.

Планируемый пайплайн:

1. Загрузка и нормализация датасета.
2. Построение агрегированных признаков.
3. Разделение на train/test до любых операций балансировки.
4. SMOTE только на train-части.
5. Fit scaler только на train-части.
6. Сохранение схемы признаков.
7. Сравнение MLP, RandomForest, XGBoost/LightGBM и AutoEncoder.
8. Экспорт выбранной модели в ONNX.

Сырые события не являются готовыми признаками для модели. Для обучения используются агрегированные window-level и flow-level датасеты:

```text
raw events -> normalized events -> feature extraction -> windows.csv / flows.csv -> training
```

Первый каталог признаков находится в `filin/ml/features/feature_catalog.yaml`. Сборщики датасетов:

```powershell
python filin/ml/features/build_windows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/windows_v0_1.csv --window-seconds 60

python filin/ml/features/build_flows_dataset.py --manifest filin/lab/output/scenario_manifest.yaml --events filin/lab/output/normalized_events.jsonl --output filin/lab/output/datasets/flows_v0_1.csv
```

`scenario_id`, `run_sequence` и planned time используются только для разметки и анализа, но не являются входными признаками модели.
