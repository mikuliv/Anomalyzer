from app.core.schemas import NetworkEvent, PredictionResult
from app.mitre.mapper import map_class_to_mitre
from app.ml.features import extract_features


class InferenceService:
    """Прототип сервиса инференса с подготовкой к подключению ONNX Runtime и scaler."""

    class_names = {
        0: "Норма",
        1: "DDoS",
        2: "Сканирование портов",
        3: "Подбор учетных данных",
        4: "Web-атака",
        5: "Botnet",
    }

    def predict(self, event: NetworkEvent) -> PredictionResult:
        features = extract_features(event)
        class_id = self._heuristic_class(event, features)
        class_name = self.class_names[class_id]
        confidence = 0.55 if class_id == 0 else 0.74
        risk_level = self._risk_for_class(class_name, confidence)
        top_features = sorted(features, key=features.get, reverse=True)[:5]

        return PredictionResult(
            predicted_class=class_id,
            class_name=class_name,
            confidence=confidence,
            risk_level=risk_level,
            top_features=top_features,
            mitre_candidates=map_class_to_mitre(class_name),
        )

    def _heuristic_class(self, event: NetworkEvent, features: dict[str, float]) -> int:
        if features.get("syn_rate", 0) > 500 or features.get("packets", 0) > 10000:
            return 1
        if features.get("unique_destination_ports", 0) > 20:
            return 2
        if features.get("failed_logins", 0) > 5:
            return 3
        if event.destination_port in {80, 443} and features.get("http_error_rate", 0) > 0.4:
            return 4
        if features.get("periodic_beacon_score", 0) > 0.7:
            return 5
        return 0

    def _risk_for_class(self, class_name: str, confidence: float):
        if class_name in {"DDoS", "Подбор учетных данных", "Web-атака"} and confidence >= 0.7:
            return "high"
        if class_name in {"Сканирование портов", "Botnet"}:
            return "medium"
        return "low"
