from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.core.schemas import IncidentCreateRequest, RuleValidationRequest, SigmaGenerateRequest
from app.incidents.service import IncidentService
from app.ml.inference import InferenceService
from app.rules.validator import validate_rule
from app.sigma.generator import SigmaGenerator
from app.storage.repository import InMemoryIncidentRepository

app = FastAPI(title=settings.project_name, version=settings.api_version)

repository = InMemoryIncidentRepository()
inference_service = InferenceService()
incident_service = IncidentService(repository=repository, inference_service=inference_service)
sigma_generator = SigmaGenerator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.project_name}


@app.post("/api/v1/predict")
def predict(request: IncidentCreateRequest):
    return inference_service.predict(request.event)


@app.post("/api/v1/incidents")
def create_incident(request: IncidentCreateRequest):
    return incident_service.create_from_event(request.event, source=request.source)


@app.get("/api/v1/incidents/{incident_id}")
def get_incident(incident_id: str):
    incident = incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Инцидент не найден")
    return incident


@app.post("/api/v1/sigma/generate")
def generate_sigma(request: SigmaGenerateRequest):
    incident = incident_service.get(request.incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Инцидент не найден")
    return sigma_generator.generate(incident)


@app.post("/api/v1/rules/validate")
def validate_sigma_rule(request: RuleValidationRequest):
    return validate_rule(request.rule)
