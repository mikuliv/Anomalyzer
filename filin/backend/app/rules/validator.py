from uuid import uuid4

from app.core.schemas import RuleValidationResult


def validate_rule(rule: str) -> RuleValidationResult:
    normalized = rule.lower()
    if "condition:" not in normalized or "detection:" not in normalized:
        status = "rejected"
        matched_events = 0
        recommendation = "Добавить разделы Sigma detection и condition перед проверкой на стенде."
        notes = ["Структура правила неполная."]
    elif "experimental" in normalized:
        status = "needs_review"
        matched_events = 3
        recommendation = "Проверить кандидат на лабораторных событиях Zeek или Suricata и разобрать ложные срабатывания."
        notes = ["Экспериментальные правила требуют проверки аналитиком и воспроизведения на стенде."]
    else:
        status = "approved"
        matched_events = 5
        recommendation = "Правило можно использовать в демонстрационных сценариях после описания покрытия проверками."
        notes = ["Прототип валидатора не выявил очевидного источника ложных срабатываний."]

    return RuleValidationResult(
        rule_id=f"validation-{uuid4().hex[:8]}",
        status=status,
        matched_events=matched_events,
        false_positive_notes=notes,
        recommendation=recommendation,
    )
