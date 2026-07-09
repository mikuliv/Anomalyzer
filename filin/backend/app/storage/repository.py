from typing import Protocol

from app.core.schemas import IncidentCard


class IncidentRepository(Protocol):
    def save(self, incident: IncidentCard) -> None:
        ...

    def get(self, incident_id: str) -> IncidentCard | None:
        ...


class InMemoryIncidentRepository:
    def __init__(self) -> None:
        self._items: dict[str, IncidentCard] = {}

    def save(self, incident: IncidentCard) -> None:
        self._items[incident.incident_id] = incident

    def get(self, incident_id: str) -> IncidentCard | None:
        return self._items.get(incident_id)
