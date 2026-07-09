from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(title=settings.project_name, version=settings.api_version)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.project_name}
