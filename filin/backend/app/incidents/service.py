from datetime import UTC, datetime
from uuid import uuid4

from app.core.schemas import IncidentCard, NetworkEvent
from app.incidents.risk import normalize_risk
from app.ml.inference import InferenceService
from app.sigma.generator import SigmaGenerator
from app.storage.repository import IncidentRepository


class IncidentService:
    def __init__(self, repository: IncidentRepository, inference_service: InferenceService) -> None:
        self.repository = repository
        self.inference_service = inference_service
        self.sigma_generator = SigmaGenerator()

    def create_from_event(self, event: NetworkEvent, source: str) -> IncidentCard:
        prediction = self.inference_service.predict(event)
        risk_level = normalize_risk(prediction)
        incident = IncidentCard(
            incident_id=f"inc-{uuid4().hex[:12]}",
            created_at=datetime.now(UTC).isoformat(),
            source=source,
            prediction=prediction,
            risk_level=risk_level,
            mitre_candidates=prediction.mitre_candidates,
            sigma_rule=None,
            status="new",
            analyst_notes=[],
            event=event,
        )
        incident.sigma_rule = self.sigma_generator.generate(incident).rule
        self.repository.save(incident)
        return incident

    def get(self, incident_id: str) -> IncidentCard | None:
        return self.repository.get(incident_id)
