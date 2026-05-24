import sys
sys.path.append('.')
from ai_agent.auto_engine import apply_auto_actions_snapshot
import pandas as pd

def make_df():
    rng = pd.date_range('2020-01-01', periods=30, freq='D')
    return pd.DataFrame({'date': rng, 'value': range(30)})

state = {"agent_requirements": [{"text": "Use time series Holt Winters multiplicative model with seasonal 12"}]}
from ai_agent.copilot_utils import infer_time_series_configuration
print('INFERRED TS CONFIG:', infer_time_series_configuration(state['agent_requirements'][0]['text']))
changes, activities = apply_auto_actions_snapshot(state, 3, make_df())
print('CHANGES:', changes)
print('ACTIVITIES:', activities)
