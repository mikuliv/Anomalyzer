from app.core.schemas import PredictionResult, RiskLevel


def normalize_risk(prediction: PredictionResult) -> RiskLevel:
    if prediction.risk_level == "critical":
        return "critical"
    if prediction.confidence >= 0.85 and prediction.class_name != "Normal":
        return "critical"
    if prediction.confidence >= 0.7 and prediction.class_name != "Normal":
        return "high"
    if prediction.class_name != "Normal":
        return "medium"
    return "low"
