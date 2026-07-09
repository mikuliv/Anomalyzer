from app.core.schemas import PredictionResult, RiskLevel


def normalize_risk(prediction: PredictionResult) -> RiskLevel:
    if prediction.risk_level == "critical":
        return "critical"
    if prediction.confidence >= 0.85 and prediction.class_name != "Норма":
        return "critical"
    if prediction.confidence >= 0.7 and prediction.class_name != "Норма":
        return "high"
    if prediction.class_name != "Норма":
        return "medium"
    return "low"
