"""
memory.py — Conversation memory orchestrator.

Bridges database.py (storage) + summarizer.py (compression) into one
simple interface used by chatbot.py and app.py.

Flow on every message:
  1. Load all messages for this conversation from DB
  2. If conversation is long, summarize old messages (cached summary in DB)
  3. Return (summary, recent_messages) ready to send to Groq
  4. After the response, save both user + assistant messages to DB
"""

from __future__ import annotations
import streamlit as st
from src import database as db
from src.summarizer import build_context_with_summary, needs_summarization


def get_session_key() -> str:
    """
    Unique key per browser session. Streamlit gives each browser tab
    a stable session_id for the lifetime of that tab/connection.
    """
    if "session_key" not in st.session_state:
        import uuid
        st.session_state.session_key = str(uuid.uuid4())
    return st.session_state.session_key


def init_conversation() -> tuple[str, str]:
    """
    Call once at app startup. Returns (user_id, conversation_id).
    Reuses an existing conversation for this browser session if found,
    otherwise creates a fresh user + conversation.
    """
    session_key = get_session_key()
    user_id = db.get_or_create_user(session_key)

    conversation_id = db.get_latest_conversation(user_id)
    if not conversation_id:
        conversation_id = db.create_conversation(user_id)

    return user_id, conversation_id


def load_history_for_ui(conversation_id: str) -> list[dict]:
    """
    Returns full message history formatted for st.session_state.messages
    (used to repopulate the chat UI on page refresh).
    """
    rows = db.get_messages(conversation_id)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_llm_context(conversation_id: str) -> tuple[list[dict], str | None]:
    """
    Returns (recent_messages, summary) ready to pass to Groq.
    Automatically summarizes old messages if the conversation has grown long.
    """
    all_rows = db.get_messages(conversation_id)
    all_messages = [{"role": r["role"], "content": r["content"]} for r in all_rows]

    existing_summary = db.get_conversation_summary(conversation_id)

    if not needs_summarization(len(all_messages)):
        return all_messages, existing_summary

    recent, new_summary = build_context_with_summary(all_messages, existing_summary)

    # Persist the updated summary so we don't re-summarize next turn
    if new_summary and new_summary != existing_summary:
        db.update_conversation_summary(conversation_id, new_summary)

    return recent, new_summary


def save_turn(
    conversation_id: str,
    user_text: str,
    assistant_text: str,
    nlp_meta: dict,
) -> None:
    """Persists one full user+assistant turn to the database."""
    db.save_message(conversation_id, "user", user_text)
    db.save_message(
        conversation_id,
        "assistant",
        assistant_text,
        intent=nlp_meta.get("intent"),
        sentiment=nlp_meta.get("sentiment"),
        entities=nlp_meta.get("entities"),
    )


def clear_conversation(conversation_id: str) -> None:
    db.delete_conversation_messages(conversation_id)