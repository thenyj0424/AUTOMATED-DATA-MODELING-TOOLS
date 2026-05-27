from datetime import datetime
import time

import streamlit as st

from ai_agent.copilot_utils import (
    build_copilot_context,
    condense_user_facing_text,
    build_meta_help_reply,
    build_dataset_analysis_context,
    build_unsupported_tool_reply,
    is_dataset_question,
    is_meta_help_request,
    is_requirement_message,
    is_unsupported_statistical_request,
    is_statistical_diagnostic_request,
    strip_internal_plan_payload,
    sanitize_user_facing_text,
    update_conversation_memory_from_assistant,
    update_conversation_memory_from_user,
)
from ai_agent.llm_client import call_groq, groq_token_loaded, explain_groq_error, classify_task_route
from ai_agent.llm_client import DEFAULT_REASONER_MODEL


def render_chat_dock() -> None:
    """Render a compact sidebar chat panel for recent messages and prompts."""
    with st.sidebar.expander("AI Chat", expanded=False):
        messages = st.session_state.get("agent_chat_messages", [])[-20:]
        if not messages:
            st.caption("Ask for guidance on the current step.")
        for m in messages:
            role = m.get("role", "assistant")
            content = str(m.get("content", ""))
            prefix = "You" if role == "user" else "Auto AI"
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

            update_conversation_memory_from_user(user_text)

            msgs = st.session_state.get("agent_chat_messages", [])
            msgs.append({"role": "user", "content": user_text})
            st.session_state["agent_chat_messages"] = msgs[-50:]

            activity = st.session_state.get("agent_activity", [])
            activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": f"User: {user_text[:80]}"})
            st.session_state["agent_activity"] = activity[:50]

            if groq_token_loaded():
                # If the message is a requirement, record it and acknowledge — do not call the LLM.
                if is_requirement_message(user_text):
                    reqs = st.session_state.get("agent_requirements", [])
                    reqs.insert(0, {"text": user_text, "time": datetime.now().isoformat()})
                    st.session_state["agent_requirements"] = reqs[:50]
                    msgs = st.session_state.get("agent_chat_messages", [])
                    # Parse and apply the requirement immediately so the UI and auto-engine honor it.
                    from ai_agent.copilot_utils import apply_requirement_to_state
                    applied = apply_requirement_to_state(user_text)
                    for a in applied:
                        activity = st.session_state.get("agent_activity", [])
                        activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": a})
                        st.session_state["agent_activity"] = activity[:50]
                    applied_any = any(str(note).lower().startswith("applied requirement") for note in applied)
                    failed_notes = [note for note in applied if str(note).lower().startswith("requirement not applied")]
                    if failed_notes:
                        requirement_reply = str(failed_notes[0])
                    elif applied_any:
                        requirement_reply = "Noted. Requirement applied."
                    else:
                        requirement_reply = "I could not apply that requirement. Please specify a supported option."
                    msgs.append({"role": "assistant", "content": requirement_reply})
                    st.session_state["agent_chat_messages"] = msgs[-50:]
                    update_conversation_memory_from_assistant(requirement_reply)
                    activity = st.session_state.get("agent_activity", [])
                    activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": "Requirement recorded via chat."})
                    st.session_state["agent_activity"] = activity[:50]
                    return

                if is_unsupported_statistical_request(user_text):
                    reply = build_unsupported_tool_reply(user_text)
                    msgs = st.session_state.get("agent_chat_messages", [])
                    msgs.append({"role": "assistant", "content": reply})
                    st.session_state["agent_chat_messages"] = msgs[-50:]
                    update_conversation_memory_from_assistant(reply)
                    activity = st.session_state.get("agent_activity", [])
                    activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": "Unsupported statistical request redirected to available tools."})
                    st.session_state["agent_activity"] = activity[:50]
                    return

                with st.spinner("AI thinking..."):
                    context = build_copilot_context(
                        st.session_state.get("step", 0),
                        st.session_state.get("df"),
                        st.session_state.get("summary"),
                    )
                    if is_meta_help_request(user_text):
                        reply = build_meta_help_reply(user_text, st.session_state.get("agent_goal", ""))
                    elif is_dataset_question(user_text, st.session_state.get("step", 0), st.session_state.get("df"), st.session_state.get("summary")) or is_statistical_diagnostic_request(user_text):
                        dataset_context = build_dataset_analysis_context(
                            st.session_state.get("step", 0),
                            st.session_state.get("df"),
                            st.session_state.get("summary"),
                            user_text,
                        )
                        prompt = (
                            "You are Auto AI. Give the final recommendation first. "
                            "Keep the reply to 1-3 short sentences. Do not describe your reasoning process or mention internal analysis. "
                            "Use the tool-backed EDA bundle only to support the final answer, and mention missingness, stationarity, or target cues only if they directly help the result. "
                            "Do not mention model labels or phrases like 'the user wants'.\n"
                            f"Context:\n{context}\n\n"
                            f"Tool-backed EDA bundle:\n{dataset_context['prompt_context']}\n\n"
                            f"User question: {user_text}"
                        )
                        activity = st.session_state.get("agent_activity", [])
                        activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": "Auto AI used perform_eda for the dataset question."})
                        st.session_state["agent_activity"] = activity[:50]
                        reply = call_groq(
                            prompt,
                            max_new_tokens=240,
                            task_type="reasoning",
                            context=dataset_context["prompt_context"],
                            force_model=DEFAULT_REASONER_MODEL,
                        ) or ""
                    else:
                        selected_model_name = classify_task_route(user_text, context)
                        prompt = (
                            "You are Auto AI. Answer with the result first. "
                            "Keep the reply short, direct, and action-oriented. Do not explain your thinking unless the user asks for it. "
                            "Do not mention which model is speaking and do not use self-interpretations like 'the user wants'.\n"
                            f"Context:\n{context}\n\n"
                            f"User request: {user_text}"
                        )
                        reply = call_groq(
                            prompt,
                            max_new_tokens=180,
                            task_type="chat",
                            context=context,
                            force_model=selected_model_name,
                        ) or ""
                if reply.startswith("ERROR:"):
                    activity = st.session_state.get("agent_activity", [])
                    activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "error", "message": reply})
                    st.session_state["agent_activity"] = activity[:50]
                    reply = f"{explain_groq_error(reply)} I can still guide you using current workflow context."
                elif not reply.strip():
                    reply = "I do not have a generated response right now, but I can still guide your next workflow step."
                reply = condense_user_facing_text(strip_internal_plan_payload(sanitize_user_facing_text(reply)))
                msgs = st.session_state.get("agent_chat_messages", [])
                msgs.append({"role": "assistant", "content": reply})
                st.session_state["agent_chat_messages"] = msgs[-50:]
                update_conversation_memory_from_assistant(reply)
                activity = st.session_state.get("agent_activity", [])
                activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "info", "message": "Auto AI replied to chat."})
                st.session_state["agent_activity"] = activity[:50]
            else:
                msgs = st.session_state.get("agent_chat_messages", [])
                missing_llm_reply = "LLM not configured. Set GROQ_API_KEY to enable responses."
                msgs.append({"role": "assistant", "content": missing_llm_reply})
                st.session_state["agent_chat_messages"] = msgs[-50:]
                update_conversation_memory_from_assistant(missing_llm_reply)
                activity = st.session_state.get("agent_activity", [])
                activity.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "ts": time.time(), "level": "warn", "message": "LLM not configured; no reply."})
                st.session_state["agent_activity"] = activity[:50]
