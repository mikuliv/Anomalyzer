from app.core.schemas import MitreCandidate
from app.mitre.techniques import MITRE_BY_CLASS


def map_class_to_mitre(class_name: str) -> list[MitreCandidate]:
    return MITRE_BY_CLASS.get(class_name, [])
