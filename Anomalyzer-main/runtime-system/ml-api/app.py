from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Response
import onnxruntime as ort
import numpy as np
import joblib
import json


# Инициализация
app = FastAPI()
session = ort.InferenceSession("mlp_cicids.onnx")
scaler = joblib.load("scaler.pkl")
classes = np.load("classes.npy")

# Список нужных признаков (тот же порядок, что при обучении)
REQUIRED_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Bulk Rate Avg", "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

class LogEntry(BaseModel):
    log: dict

@app.post("/predict")
async def predict(log_entry: LogEntry):
    try:
        log = log_entry.log
        features = [float(log.get(f, 0.0)) for f in REQUIRED_FEATURES]
        X_scaled = scaler.transform([features]).astype(np.float32)
        outputs = session.run(None, {"input": X_scaled})
        predicted_class = int(np.argmax(outputs[0][0]))
        result = {
            "predicted_class": str(predicted_class),
            "class_name": str(classes[predicted_class])
        }
        # Возвращаем строку JSON, а не dict!
        return Response(content=json.dumps(result), media_type="application/json")
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), media_type="application/json")
