from app.core.schemas import MitreCandidate


MITRE_BY_CLASS: dict[str, list[MitreCandidate]] = {
    "DDoS": [
        MitreCandidate(
            tactic="impact",
            technique_id="T1498",
            technique_name="Network Denial of Service",
            confidence=0.72,
            explanation="High packet or SYN rate can indicate network-layer denial of service activity.",
        )
    ],
    "PortScan": [
        MitreCandidate(
            tactic="discovery",
            technique_id="T1046",
            technique_name="Network Service Discovery",
            confidence=0.76,
            explanation="Many destination ports against one or more hosts matches service discovery behavior.",
        )
    ],
    "Brute Force": [
        MitreCandidate(
            tactic="credential-access",
            technique_id="T1110",
            technique_name="Brute Force",
            confidence=0.78,
            explanation="Repeated failed authentication attempts are consistent with brute force activity.",
        )
    ],
    "Web Attack": [
        MitreCandidate(
            tactic="initial-access",
            technique_id="T1190",
            technique_name="Exploit Public-Facing Application",
            confidence=0.68,
            explanation="Suspicious web errors or payload indicators can point to public-facing application exploitation.",
        )
    ],
    "Botnet": [
        MitreCandidate(
            tactic="command-and-control",
            technique_id="T1071",
            technique_name="Application Layer Protocol",
            confidence=0.55,
            explanation="Periodic beacon-like traffic is only a cautious candidate and requires confirmation with DNS, HTTP or TLS logs.",
        )
    ],
}
