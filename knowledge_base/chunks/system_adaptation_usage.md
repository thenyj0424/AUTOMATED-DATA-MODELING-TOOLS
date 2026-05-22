# System Adaptation Usage

This chunk defines how to map requests into valid session-key changes.

JSON output schema:
```
{"changes": {"key": value}, "activities": ["short note"], "priority_source": "user|kb|dataset"}
```

Blocked keys (never set):
- step
- cleaning_confirmed
- analysis_ready

Step-2 cleaning keys:
- missing_strategy: impute | drop
- numeric_impute_strategy: median | mean
- impute_categorical: true | false

Step-3 modelling keys:
- problem_type: classification | regression | time_series
- model_family: Statistical | ML
- model_name: exact supported model string
- time_series_model: ARIMA | SARIMA | Holt-Winters
- time_series_mode: Auto ARIMA | Manual
- hw_trend: add | mul | None
- hw_seasonal_periods: integer 2-24
- sarima_m: integer 2-24
- metric_label: valid metric for current problem type
- feature_selection: None | Stepwise | SelectKBest | RFE | Model-based
- tuning_method: None | Grid Search | Manual | TPE (Optuna)
- proceed_model: true | false

Requirement precedence:
- Explicit valid user requirement overrides default recommendation.
- If user does not specify details, pick best valid default for current step.

Model-family alignment:
- Logistic Regression, Linear Regression, Ridge, Lasso -> Statistical
- Decision Tree, Random Forest, Gradient Boosting, SVM, SVR, KNN -> ML

Efficiency:
- Return only needed keys for current step.
- Keep activities concise and operational.
