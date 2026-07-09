import torch
import torch.nn as nn
from train_mlp_stable import BetterMLP  # импорт класса модели из файла обучения
import os

# === Конфигурация ===
MODEL_PATH = "mlp_cicids_full.pt"
ONNX_PATH = "mlp_cicids.onnx"
INPUT_SIZE = 78
NUM_CLASSES = 15

# === Разрешаем десериализацию класса MLP ===
torch.serialization.add_safe_globals({'MLP': BetterMLP})

# === Загружаем модель ===
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# === Заглушка для входного тензора ===
dummy_input = torch.randn(1, INPUT_SIZE)

# === Экспорт в ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ Модель успешно экспортирована в ONNX: {ONNX_PATH}")
