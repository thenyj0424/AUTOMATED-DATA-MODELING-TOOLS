"""Demo script for the auto action engine.
Run from project root: `python scripts/demo_auto_engine.py`
This script runs the auto_engine on a synthetic DataFrame and prints the suggested changes.
"""
from ai_agent.auto_engine import apply_auto_actions_snapshot
import pandas as pd

# Create synthetic dataset
N = 50
rng = pd.date_range('2020-01-01', periods=N, freq='D')
df = pd.DataFrame({'date': rng, 'value': range(N)})
# inject some missing
import numpy as np

df.loc[5:8, 'value'] = np.nan

state = {}

print('=== Step 1 (EDA) ===')
changes, activities = apply_auto_actions_snapshot(state, 1, df)
print('Changes:', list(changes.keys()))
print('Activities:', activities)

print('\n=== Step 2 (Cleaning) ===')
changes, activities = apply_auto_actions_snapshot(state, 2, df)
print('Changes keys:', list(changes.keys()))
print('Activities:', activities)

print('\n=== Step 3 (Modeling) ===')
changes, activities = apply_auto_actions_snapshot(state, 3, df)
print('Changes keys:', list(changes.keys()))
print('Activities:', activities)
