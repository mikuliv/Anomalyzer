from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.core.schemas import IncidentCreateRequest
from app.incidents.service import IncidentService
from app.ml.inference import InferenceService
from app.storage.repository import InMemoryIncidentRepository

app = FastAPI(title=settings.project_name, version=settings.api_version)

repository = InMemoryIncidentRepository()
inference_service = InferenceService()
incident_service = IncidentService(repository=repository, inference_service=inference_service)


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
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
