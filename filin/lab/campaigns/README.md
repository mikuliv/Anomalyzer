# Кампании независимых laboratory runs

Кампания v0.2.3 отделяет независимые фактические Docker executions от связанных окон одного выполнения. Train и test используют разные run ID, seeds и параметры сценариев.

Запуск: `python filin/lab/campaigns/run_campaign.py --campaign filin/lab/campaigns/v0_2_3_independent_executions.yaml --output-root filin/lab/output --strict`.
