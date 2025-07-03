
# 🛡️ Anomalyzer — AI-based Anomaly Detection in Network Traffic

**Anomalyzer** is a complete open-source system designed to detect network anomalies using a neural network (MLP) and an integrated ELK stack. This project was originally developed and defended as a graduation thesis in June 2025 and is now published for community use under the GPLv3 license.

---

## 📖 Overview

Anomalyzer performs classification of network flows based on 78 numerical features extracted from the CICIDS2017 dataset. It provides a real-time prediction pipeline that combines a pre-trained MLP model (in ONNX format) with Logstash and FastAPI, enabling inference and visualization of network behavior in Kibana.

---

## 🎯 Project Goals

- ✅ Build a robust neural network (MLP) to classify various types of network traffic
- ✅ Export model to ONNX format for high-performance runtime inference
- ✅ Implement FastAPI-based RESTful API for prediction delivery
- ✅ Integrate with Logstash to accept input from structured CSV data
- ✅ Enable visualization and monitoring using Kibana dashboards
- ✅ Make the entire system portable and reproducible using Docker

---

## 🏗️ Architecture Diagram

![Architecture](docs/screenshots/Docker_Project.png)

---

## 🔬 Technologies Used

- **PyTorch** — for training the MLP model
- **ONNX** — for converting and running model in production
- **FastAPI** — for exposing prediction service via REST API
- **Scikit-learn** — for preprocessing, LabelEncoder, SMOTE
- **Elasticsearch** — storage and indexing of inference results
- **Logstash** — reading and transforming network flow records
- **Kibana** — dashboarding and real-time monitoring
- **Docker & Docker Compose** — for containerization of all components

---

## 📦 Repository Structure

```bash
Anomalyzer/
├── model-training/       # Code to train, export, and test the MLP model
├── pretrained-models/    # ONNX model, scaler, label encoder artifacts
├── runtime-system/       # Docker-compose infra: Logstash, ml-api, etc.
├── docs/                 # Diagrams, screenshots, model graphs
└── README.md             # This file
```

---

## 🚀 Quick Start

> ⚠️ Requires Docker + Docker Compose installed

```bash
cd runtime-system
cp .env.example .env
docker-compose up --build
```

Then:
- `ml-api` will be available at `http://localhost:8000/predict`
- `logstash` reads CSV from `logstash/data/*.csv`
- `elasticsearch` stores results
- `kibana` available at `http://localhost:5601`

---

## 🧪 Sample Prediction Request

```http
POST /predict
Content-Type: application/json

{
  "features": [123, 6, 200000, 10, 8, 500, 400, ..., 0]
}
```

Response:
```json
{
  "predicted_class": "BENIGN",
  "probabilities": {
    "BENIGN": 0.992,
    "DDoS": 0.003,
    ...
  }
}
```

---

## 📊 Model Performance

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.987 |
| F1-Score   | 0.986 |
| ROC-AUC    | 0.993 |
| Precision  | 0.988 |
| Recall     | 0.984 |

---

## 🧬 Dataset Information

This project uses the **CICIDS2017** dataset provided by the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). It includes a wide range of normal and malicious traffic scenarios, annotated and processed into 78 statistical flow features.

All rights and credits for the original dataset belong to the University of New Brunswick.

---

## 📘 Author

**Ruslan Pokatilov**  
Specialist in Information Security (10.02.05)  
Graduated with distinction in 2025  
Location: Russia 🇷🇺

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0**.

> You are free to use, share, and modify this project under the same license.  
> Full text: https://www.gnu.org/licenses/gpl-3.0.html
