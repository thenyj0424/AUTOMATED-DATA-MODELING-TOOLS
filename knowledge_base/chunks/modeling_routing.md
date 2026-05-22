---
title: Modeling Routing
tags: [routing, models, orchestration, validation]
priority: 8
summary: Routing and fallback policy for router, reasoner, and reviewer models.
---

# Modeling Routing

Use router model for:
- intent classification
- lightweight chat routing
- simple extraction

Use reasoner model for:
- main workflow planning
- KB-grounded action planning
- structured JSON plan generation

Use reviewer model for:
- high-risk validation
- uncertainty checks
- backup fallback when router/reasoner fail, rate-limit, or quota-limit

Fallback policy:
- If primary selected model fails, retry through fallback chain.
- Prefer reviewer as immediate backup for reasoning-critical requests.
- Log fallback in activity notes when possible.