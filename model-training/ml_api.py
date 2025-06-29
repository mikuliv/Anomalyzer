from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load
import onnxruntime as ort

app = FastAPI(title="MLP CICIDS ONNX API")

# Загрузка модели, скалера и меток классов
scaler = load("scaler.pkl")
classes = np.load("classes.npy", allow_pickle=True)
session = ort.InferenceSession("mlp_cicids.onnx")

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    if len(data.features) != 78:
        raise HTTPException(status_code=400, detail="Нужно передать ровно 78 признаков.")

    # Подготовка входных данных
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X).astype(np.float32)

    # Инференс ONNX модели
    inputs = {session.get_inputs()[0].name: X_scaled}
    outputs = session.run(None, inputs)
    
    # Получение индекса и названия предсказанного класса
    pred_index = int(np.argmax(outputs[0], axis=1)[0])
    pred_label = str(classes[pred_index])

    # Форматирование логитов в словарь
    probs = outputs[0][0]
    prob_dict = {str(classes[i]): float(prob) for i, prob in enumerate(probs)}

    return {
        "predicted_class": pred_label,
        "probabilities": prob_dict
    }
