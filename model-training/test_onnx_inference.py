import onnxruntime as ort
import numpy as np
import pandas as pd
from joblib import load

# Пути к файлам
ONNX_MODEL_PATH = "mlp_cicids.onnx"
SCALER_PATH = "scaler.pkl"
CLASSES_PATH = "classes.npy"
CSV_PATH = "CICIDS2017_FULL_prepared.csv"

# Загрузка модели и утилит
print("📦 Загрузка модели и вспомогательных файлов...")
session = ort.InferenceSession(ONNX_MODEL_PATH)
scaler = load(SCALER_PATH)
classes = np.load(CLASSES_PATH)
print("✅ Загрузка завершена")

# Загрузка данных
print("📥 Загрузка и подготовка CSV...")
df = pd.read_csv(CSV_PATH)

# Удаление метки, если есть
if "Label" in df.columns:
    df = df.drop(columns=["Label"])

# Ввод: 1-я строка
row = df.iloc[0].values.reshape(1, -1)
row_scaled = scaler.transform(row).astype(np.float32)

# Инференс
print("🧠 Предсказание...")
inputs = {"input": row_scaled}
outputs = session.run(None, inputs)
preds = np.argmax(outputs[0], axis=1)

# Результат
print(f"🔍 Класс: {classes[preds[0]]}")
