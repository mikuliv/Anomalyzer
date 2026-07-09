from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project_name: str = "Филин"
    api_version: str = "0.1.0"
    model_path: str | None = None
    scaler_path: str | None = None


settings = Settings()
