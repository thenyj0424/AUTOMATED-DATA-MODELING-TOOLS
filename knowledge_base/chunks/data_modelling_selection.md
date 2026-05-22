---
title: Data Modelling Selection
tags: [modeling, selection, pipeline, insert-space]
priority: 10
summary: Step-3 modelling rules for problem type, family, model, feature selection, and tuning.
---

# Data Modelling Selection

Use this chunk for Step 3 decisions only.
Preserve explicit valid user requests. If unspecified, choose valid defaults.

Problem type rules:
- Categorical target -> classification.
- Numeric target -> regression unless forecasting intent + datetime support exists.
- Forecasting intent + datetime -> time_series.

Time-series request rules:
- If user explicitly requests ARIMA/SARIMA/Holt-Winters, preserve that model choice.
- "Holt-Winters multiplicative seasonal 12" -> problem_type=time_series, time_series_model=Holt-Winters, hw_trend=mul, hw_seasonal_periods=12.
- For SARIMA with seasonal 12 -> set sarima_m=12.
- Do not drift to Auto ARIMA when user explicitly requested Holt-Winters.

Model family rules:
- Statistical family for simpler interpretable baselines.
- ML family for non-linear patterns or explicit performance-focused requests.
- Always align model_family with model_name.

Supported models (exact strings):
- Statistical classification: Logistic Regression
- Statistical regression: Linear Regression, Ridge, Lasso
- ML classification: Decision Tree, Random Forest, Gradient Boosting, SVM, KNN
- ML regression: Decision Tree, Random Forest, Gradient Boosting, SVR, KNN

Request mapping examples:
- "use knn" -> model_family=ML, model_name=KNN
- "use random forest" -> model_family=ML, model_name=Random Forest
- "use logistic" -> model_family=Statistical, model_name=Logistic Regression

Robust request understanding:
- Accept common typo variants and compact forms (for example "decisiontree", "decison tree", "randomforest", "k nearest").
- Map to nearest valid supported model name when confidence is high.
- If ambiguous, prefer the latest explicit user request and log a short clarification activity.

Change-of-mind policy:
- Latest user intent wins over older requirements.
- Apply latest valid model request even if earlier messages requested a different model.
- Keep in-framework constraints (valid model list and family alignment) while applying the latest intent.

Feature selection rules:
- If features are few, prefer None.
- If statistical family and feature count is moderate/high, allow Stepwise.
- If ML with many features, allow SelectKBest.

Tuning rules:
- If user requests quick run or data is small/simple, prefer no heavy tuning first.
- If user requests best performance and model is ML, allow Grid Search.

Unspecified requests:
- If user does not mention model/family, choose valid defaults from dataset + goal.
- Record a short activity note describing why the default was chosen.

Never output unsupported model names.