---
title: Model Evaluations
tags: [evaluation, metrics, validation, insert-space]
priority: 10
summary: Step-4 rules for selecting the best result and recommending the next valid action.
---

# Model Evaluations

Use this chunk for Step 4 only.

Evaluation rules:
- Classification: prioritize selected metric, then weighted F1 and confusion behavior.
- Regression: consider R2 with MAE/RMSE together.
- Time-series: consider forecast error and residual behavior.
- Compare baseline and tuned outputs when both exist.
- Prefer tuned model only when gain is meaningful and consistent.

Recommendation rules:
- If result is strong: keep current configuration.
- If weak due to data quality: suggest cleaning updates first.
- If weak due to model mismatch: suggest family/model change.
- If unstable: suggest simpler model or reduced feature complexity.

Efficiency rules:
- Keep explanation short: metric -> interpretation -> next action.
- Avoid long narrative when one clear recommendation is sufficient.