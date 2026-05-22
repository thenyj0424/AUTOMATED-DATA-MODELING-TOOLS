---
title: System Adaptation Protocol
tags: [protocol, system, adaptation, recommendation, workflow]
priority: 20
summary: Core policy for reliable, low-cost recommendations inside the fixed app workflow.
---

# System Adaptation Protocol

The system structure is fixed. Do not invent new UI steps, new controls, or unsupported automation.
Only propose actions that map to existing session keys and supported model options.

Use two signals together:
- the user request
- the uploaded dataset profile

When the user request is vague, incomplete, or broad, choose the best valid default for the current step and dataset.

Decision hierarchy:
1. Respect explicit user requirements when valid.
2. Enforce dataset and workflow constraints.
3. If unspecified, choose an in-framework default automatically.
4. Keep output concise and executable.

Decision rules:
- If target is categorical, treat as classification.
- If target is numeric, treat as regression unless forecasting intent is explicit.
- If datetime exists and forecasting is requested, use time-series flow.
- If data quality is poor, prioritize cleaning before advanced modelling.

Model control rules for auto-mode:
- Use exact model names from app config lists only.
- Keep explicit model requests exactly when valid (for example KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, SVR).
- Ensure model family aligns with model name.
- Third model acts as backup when router or reasoner fails, rate-limits, or quota-limits.
- If requested model cannot run, choose nearest valid model and log the reason in activities.

Efficiency rules for free tier:
- Prefer minimal context needed for correct decisions.
- Retrieve high-priority step-specific chunks first.
- Avoid broad generic explanations in generated plans.
- Keep activities short and operational.

Safety rules:
- Never set control-flow keys in AI plan: step, cleaning_confirmed, analysis_ready.
- Never propose unknown session keys.

Goal: high-accuracy decisions within the fixed system at low token cost.