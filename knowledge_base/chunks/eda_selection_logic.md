---
title: EDA Selection Logic
tags: [eda, selection, workflow, insert-space]
priority: 10
summary: Step-1 EDA rules with explicit key mapping and low-cost defaults.
---

# EDA Selection Logic

Use this chunk for Step 1 decisions.
Only set supported EDA keys and only when needed.

Supported Step-1 keys:
- eda_show_summary
- eda_show_corr
- eda_show_basic
- eda_show_missing
- eda_show_target_dist
- eda_show_target_rel
- eda_show_pairplot
- eda_show_outliers
- eda_target_col
- eda_target_feature
- eda_numeric_column

Selection policy:
- If user asks for a specific EDA view, enable that view first.
- If user does not specify, enable: summary + missingness first.
- Enable correlation only if at least two numeric columns exist.
- Enable target distribution/relationships only if a valid target is known.
- Enable pairplot only for smaller feature sets to avoid heavy output.
- Enable outlier summary when numeric columns exist.

Target policy:
- Prefer a categorical target for classification-oriented EDA.
- Else fallback to a meaningful numeric target.

Efficiency policy:
- Do not enable all plots by default on wide datasets.
- Prefer one or two high-signal visuals first, then expand only if asked.