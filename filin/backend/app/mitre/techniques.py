from app.core.schemas import MitreCandidate


MITRE_BY_CLASS: dict[str, list[MitreCandidate]] = {
    "DDoS": [
        MitreCandidate(
            tactic="impact",
            technique_id="T1498",
            technique_name="Network Denial of Service",
            confidence=0.72,
            explanation="Высокая интенсивность пакетов или SYN-запросов может указывать на сетевой отказ в обслуживании.",
        )
    ],
    "Сканирование портов": [
        MitreCandidate(
            tactic="discovery",
            technique_id="T1046",
            technique_name="Network Service Discovery",
            confidence=0.76,
            explanation="Большое число портов назначения для одного или нескольких узлов соответствует разведке сетевых сервисов.",
        )
    ],
    "Подбор учетных данных": [
        MitreCandidate(
            tactic="credential-access",
            technique_id="T1110",
            technique_name="Brute Force",
            confidence=0.78,
            explanation="Повторяющиеся неуспешные попытки аутентификации соответствуют подбору учетных данных.",
        )
    ],
    "Web-атака": [
        MitreCandidate(
            tactic="initial-access",
            technique_id="T1190",
            technique_name="Exploit Public-Facing Application",
            confidence=0.68,
            explanation="Подозрительные web-ошибки или признаки полезной нагрузки могут указывать на эксплуатацию публичного приложения.",
        )
    ],
    "Botnet": [
        MitreCandidate(
            tactic="command-and-control",
            technique_id="T1071",
            technique_name="Application Layer Protocol",
            confidence=0.55,
            explanation="Периодический beacon-трафик является только осторожным кандидатом и требует подтверждения по DNS, HTTP или TLS-логам.",
        )
    ],
}
