from uuid import uuid4

from app.core.schemas import RuleValidationResult


def validate_rule(rule: str) -> RuleValidationResult:
    normalized = rule.lower()
    if "condition:" not in normalized or "detection:" not in normalized:
        status = "rejected"
        matched_events = 0
        recommendation = "Add Sigma detection and condition sections before lab validation."
        notes = ["The rule structure is incomplete."]
    elif "experimental" in normalized:
        status = "needs_review"
        matched_events = 3
        recommendation = "Run the candidate against Zeek or Suricata lab events and review false positives."
        notes = ["Experimental rules require analyst review and replay on the validation stand."]
    else:
        status = "approved"
        matched_events = 5
        recommendation = "Rule can be used in demo scenarios after documenting test coverage."
        notes = ["No obvious false-positive pattern found in the prototype validator."]

    return RuleValidationResult(
        rule_id=f"validation-{uuid4().hex[:8]}",
        status=status,
        matched_events=matched_events,
        false_positive_notes=notes,
        recommendation=recommendation,
    )
