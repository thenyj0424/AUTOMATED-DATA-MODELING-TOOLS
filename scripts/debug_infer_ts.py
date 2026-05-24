from ai_agent.copilot_utils import infer_time_series_configuration

texts = [
    "Use time series Holt Winters multiplicative model with seasonal 12",
    "Forecast this series",
    "Holt-Winters additive seasonal 4",
    "Use KNN model for gender classification.",
]
for t in texts:
    print('INPUT:', t)
    print(infer_time_series_configuration(t))
