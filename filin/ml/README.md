# Filin ML

Раздел содержит заготовки для обучения, оценки и экспорта моделей обнаружения сетевых аномалий.

Планируемый pipeline:

1. Загрузка и нормализация датасета.
2. Train/test split до любых операций балансировки.
3. SMOTE только на train-части.
4. Fit scaler только на train-части.
5. Сохранение feature schema.
6. Сравнение MLP, RandomForest, XGBoost/LightGBM и AutoEncoder.
7. Экспорт выбранной модели в ONNX.
