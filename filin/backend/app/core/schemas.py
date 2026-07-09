from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high", "critical"]


class NetworkEvent(BaseModel):
    source_ip: str | None = None
    destination_ip: str | None = None
    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str | None = None
    duration: float | None = Field(default=None, ge=0)
    bytes_in: int | None = Field(default=None, ge=0)
    bytes_out: int | None = Field(default=None, ge=0)
    packets: int | None = Field(default=None, ge=0)
    flags: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class MitreCandidate(BaseModel):
    tactic: str
    technique_id: str
    technique_name: str
    confidence: float = Field(ge=0, le=1)
    explanation: str


class PredictionResult(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    top_features: list[str]
    mitre_candidates: list[MitreCandidate]


class IncidentCreateRequest(BaseModel):
    source: str = "api"
    event: NetworkEvent


class IncidentCard(BaseModel):
    incident_id: str
    created_at: str
    source: str
    prediction: PredictionResult
    risk_level: RiskLevel
    mitre_candidates: list[MitreCandidate]
    sigma_rule: str | None = None
    status: Literal["new", "in_review", "closed"] = "new"
    analyst_notes: list[str] = Field(default_factory=list)
    event: NetworkEvent


class SigmaGenerateRequest(BaseModel):
    incident_id: str


class SigmaRuleDraft(BaseModel):
    rule_id: str
    title: str
    rule: str
    note: str


class RuleValidationRequest(BaseModel):
    rule: str


class RuleValidationResult(BaseModel):
    rule_id: str
    status: Literal["approved", "needs_review", "rejected"]
    matched_events: int
    false_positive_notes: list[str]
    recommendation: str
