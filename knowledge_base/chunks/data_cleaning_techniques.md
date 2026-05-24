---
title: Data Cleaning Techniques
tags: [cleaning, missing-values, outliers, preprocessing, workflow]
priority: 14
summary: Step-2 cleaning policy with deterministic defaults and explicit technique mapping.
---

# Data Cleaning Techniques

Use this chunk for Step 2 only.

Primary cleaning goals:
- Review outliers before missing-value handling.
- Resolve missing values safely.
- Keep feature meaning where possible.
- Avoid destructive cleaning unless user asks.

Supported cleaning choices:
- missing_strategy: impute or drop
- numeric_impute_strategy: median or mean
- impute_categorical: true or false

Default policy when user is unspecified:
- missing_strategy = impute
- numeric_impute_strategy = median
- impute_categorical = true

When to prefer each option:
- Use median for skewed numeric data or potential outliers.
- Use mean only when distribution is roughly symmetric.
- Use categorical imputation when missing categorical values are present.
- Use drop only when missingness is low and user prefers strict data quality.

Outlier handling policy:
- Detect outliers for review first.
- Prefer keeping outliers when the values look plausible, sparse, or domain-relevant.
- Remove outliers only when they look like clear errors, impossible values, or concentrated measurement artifacts.
- Let the user keep or remove detected outliers in the cleaning step before missing-value handling.
- If auto mode makes an outlier decision, surface it in modeling with a reversible keep/remove control.

Time-related columns:
- Preserve datetime columns by default.
- Do not convert time-series structure away unless user requests non-time-series modeling.

Activity note format:
- Keep one short line: what was chosen and why.
- Example: "Applied median/mode imputation as default because cleaning preference was not specified."
