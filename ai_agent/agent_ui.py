import streamlit as st
from ai_agent.copilot_utils import build_hybrid_step_hint, add_agent_activity
from ai_agent.llm_client import groq_token_loaded

# Note: lightweight helpers for agent UI; main render functions live in main.py

def render_agent_compact(step, df, summary):
    hint = build_hybrid_step_hint(step, df, summary)
    st.sidebar.info(f"AI hint: {hint}")
    if groq_token_loaded():
        add_agent_activity("LLM available for deeper guidance.")
