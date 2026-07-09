import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

print("✅ Импорты завершены")

class BetterMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def print_progress(epoch, batch_idx, total_batches, loss):
    percent = int((batch_idx / total_batches) * 100)
    bar = "█" * (percent // 4) + "-" * (25 - (percent // 4))
    print(f"\rEpoch {epoch+1}: |{bar}| {percent}% - Batch {batch_idx}/{total_batches} - Loss: {loss:.2f}", end="")

if __name__ == "__main__":
    print("📥 Загрузка данных...")
    df = pd.read_csv("C:/your/path/CICIDS2017_FULL_prepared.csv")
    X = df.drop(columns=["Label"])
    y = df["Label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = [str(c) for c in label_encoder.classes_]
    np.save("classes.npy", class_names)
    print("✅ Метки закодированы")

    print("🧼 Очистка NaN и Inf...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    print("✅ Очистка завершена")

    print("📊 Масштабирование...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Масштабирование завершено")

    print("⚖ Балансировка SMOTE...")
    smote = SMOTE()
    X_bal, y_bal = smote.fit_resample(X_scaled, y)
    print("✅ SMOTE завершён")

    print("🧪 Разделение train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    print("✅ Разделение завершено")

    print("📦 Подготовка тензоров...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=512, num_workers=0)
    print("✅ Тензоры готовы")

    print("🧠 Создание модели...")
    model = BetterMLP(X_train.shape[1], len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("✅ Модель готова")

    print("🚀 Начало обучения")
    epochs = 30
    losses = []
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_batches = len(train_dl)
            for i, (xb, yb) in enumerate(train_dl):
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print_progress(epoch, i + 1, total_batches, total_loss)
            losses.append(total_loss)
            print(f"  ✅ Epoch {epoch+1} complete. Total loss: {total_loss:.2f}")
        print("✅ Обучение завершено")
    except Exception as e:
        print(f"❌ Ошибка во время обучения: {e}")
        sys.exit(1)

    print("🔍 Оценка модели")
    model.eval()
    y_pred = []
    with torch.no_grad():
        for xb, _ in test_dl:
            preds = model(xb)
            y_pred.extend(torch.argmax(preds, dim=1).cpu().numpy())

    y_test_np = y_test_tensor.cpu().numpy()

    print("\n=== Classification Report ===")
    print(classification_report(y_test_np, y_pred, target_names=class_names))

    print("📊 Построение матрицы ошибок")
    cm = confusion_matrix(y_test_np, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Матрица ошибок сохранена как confusion_matrix.png")

    plt.figure()
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    print("✅ График функции потерь сохранён как loss_plot.png")

    torch.save(model.state_dict(), "mlp_cicids.pt")
    torch.save(model, "mlp_cicids_full.pt")
    print("✅ Модель сохранена как state_dict и как полная модель.")
