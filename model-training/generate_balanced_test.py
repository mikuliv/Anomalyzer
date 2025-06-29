import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === Пути к файлам ===
CSV_PATH = "C:/your/path/CICIDS2017_FULL_prepared.csv"
MODEL_PATH = "mlp_cicids.pt"
SCALER_PATH = "scaler.pkl"
CLASSES_PATH = "classes.npy"
SAMPLE_SIZE = 5000

# === Загрузка данных ===
print("📥 Загрузка выборки...")
df = pd.read_csv(CSV_PATH).sample(SAMPLE_SIZE, random_state=42)
X = df.drop(columns=["Label"])
y = df["Label"]

# === Кодирование меток ===
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(CLASSES_PATH)
y_encoded = label_encoder.transform(y)
class_names = label_encoder.classes_

# === Масштабирование признаков ===
print("📊 Масштабирование...")
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# === Определение архитектуры модели ===
print("🧠 Загрузка модели...")
class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X_scaled.shape[1], len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === Предсказания ===
print("🔍 Предсказания...")
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).numpy()

# === Анализ метрик
true_labels = np.unique(y_encoded)
print("\n=== Classification Report ===")
print(classification_report(
    y_encoded,
    y_pred,
    labels=true_labels,
    target_names=class_names[true_labels]
))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_encoded, y_pred, labels=true_labels)
print(cm)
