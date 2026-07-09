from textwrap import indent
from uuid import uuid4

from app.core.schemas import IncidentCard, SigmaRuleDraft
from app.sigma.templates import DEFAULT_LOGSOURCE, title_for_class


class SigmaGenerator:
    def generate(self, incident: IncidentCard) -> SigmaRuleDraft:
        rule_id = f"filin-{incident.incident_id}-{uuid4().hex[:6]}"
        title = title_for_class(incident.prediction.class_name)
        detection = self._build_detection(incident)
        rule = "\n".join(
            [
                f"title: {title}",
                f"id: {rule_id}",
                "status: experimental",
                "description: Черновик правила подготовлен по карточке инцидента Филин.",
                "logsource:",
                f"  product: {DEFAULT_LOGSOURCE['product']}",
                f"  category: {DEFAULT_LOGSOURCE['category']}",
                "detection:",
                indent(detection, "  "),
                "  condition: selection",
                "falsepositives:",
                "  - Требуется проверка на лабораторном стенде",
                "level: medium",
            ]
        )
        return SigmaRuleDraft(
            rule_id=rule_id,
            title=title,
            rule=rule,
            note="Sigma-правило является кандидатом и требует проверки на тестовом стенде перед применением.",
        )

    def _build_detection(self, incident: IncidentCard) -> str:
        event = incident.event
        lines = ["selection:"]
        if event.destination_ip:
            lines.append(f"  DestinationIp: {event.destination_ip}")
        if event.destination_port:
            lines.append(f"  DestinationPort: {event.destination_port}")
        if event.protocol:
            lines.append(f"  Protocol: {event.protocol}")
        lines.append(f"  FilinClass: {incident.prediction.class_name}")
        return "\n".join(lines)
