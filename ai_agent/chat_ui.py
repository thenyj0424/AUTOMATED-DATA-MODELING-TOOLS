from datetime import datetime

import streamlit as st

from ai_agent.copilot_utils import build_copilot_context
from ai_agent.llm_client import call_groq, groq_token_loaded, explain_groq_error


def render_chat_dock() -> None:
    """Render a compact sidebar chat panel for recent messages and prompts."""
    with st.sidebar.expander("AI Chat", expanded=False):
        messages = st.session_state.get("agent_chat_messages", [])[-6:]
        if not messages:
            st.caption("Ask for guidance on the current step.")
        for m in messages:
            role = m.get("role", "assistant")
            content = str(m.get("content", ""))
            prefix = "You" if role == "user" else "AI"
            st.markdown(f"**{prefix}:** {content}")

        with st.form("chat_dock_form", clear_on_submit=True):
            st.text_area(
                "Message",
                value=st.session_state.get("chat_dock_input", ""),
                key="chat_dock_input",
                placeholder="Message the AI assistant...",
                height=80,
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Send")
        if sent:
            user_text = st.session_state.get("chat_dock_input", "").strip()
            if not user_text:
                st.warning("Please enter a message before sending.")
                return

            msgs = st.session_state.get("agent_chat_messages", [])
            msgs.append({"role": "user", "content": user_text})
            st.session_state["agent_chat_messages"] = msgs[-50:]

            activity = st.session_state.get("agent_activity", [])
            activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "info", "message": f"User: {user_text[:80]}"})
            st.session_state["agent_activity"] = activity[:50]

            if groq_token_loaded():
                with st.spinner("AI thinking..."):
                    context = build_copilot_context(
                        st.session_state.get("step", 0),
                        st.session_state.get("df"),
                        st.session_state.get("summary"),
                    )
                    prompt = (
                        "Use the following workflow context to answer the user. Keep it concise and action-oriented.\n"
                        f"Context:\n{context}\n\n"
                        f"User request: {user_text}"
                    )
                    reply = call_groq(prompt, max_new_tokens=180) or ""
                if reply.startswith("ERROR:"):
                    activity = st.session_state.get("agent_activity", [])
                    activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "error", "message": reply})
                    st.session_state["agent_activity"] = activity[:50]
                    reply = f"{explain_groq_error(reply)} I can still guide you using current workflow context."
                elif not reply.strip():
                    reply = "I do not have a generated response right now, but I can still guide your next workflow step."
                msgs = st.session_state.get("agent_chat_messages", [])
                msgs.append({"role": "assistant", "content": reply})
                st.session_state["agent_chat_messages"] = msgs[-50:]
                activity = st.session_state.get("agent_activity", [])
                activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "info", "message": "AI replied to chat."})
                st.session_state["agent_activity"] = activity[:50]
            else:
                msgs = st.session_state.get("agent_chat_messages", [])
                msgs.append({"role": "assistant", "content": "LLM not configured. Set GROQ_API_KEY to enable responses."})
                st.session_state["agent_chat_messages"] = msgs[-50:]
                activity = st.session_state.get("agent_activity", [])
                activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "warn", "message": "LLM not configured; no reply."})
                st.session_state["agent_activity"] = activity[:50]
