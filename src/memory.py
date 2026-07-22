"""
memory.py — Conversation memory orchestrator.

Identity comes from the logged-in Google account (st.user.email).
Chat history follows the user across any device/browser as long as
they log in with the same Google account.
"""

from __future__ import annotations
import streamlit as st
from src import database as db
from src.summarizer import build_context_with_summary, needs_summarization


def init_conversation() -> tuple[str, str]:
    """
    Call once after confirming st.user.is_logged_in is True.
    Returns (user_id, conversation_id).
    """
    if not st.user.is_logged_in:
        raise RuntimeError("init_conversation() called before user logged in")

    user_id = db.get_or_create_user_by_email(
        email=st.user.email,
        name=getattr(st.user, "name", None),
        picture=getattr(st.user, "picture", None),
    )

    conversation_id = db.get_latest_conversation(user_id)
    if not conversation_id:
        conversation_id = db.create_conversation(user_id)

    return user_id, conversation_id


def load_history_for_ui(conversation_id: str) -> list[dict]:
    rows = db.get_messages(conversation_id)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_llm_context(conversation_id: str) -> tuple[list[dict], str | None]:
    all_rows = db.get_messages(conversation_id)
    all_messages = [{"role": r["role"], "content": r["content"]} for r in all_rows]

    existing_summary = db.get_conversation_summary(conversation_id)

    if not needs_summarization(len(all_messages)):
        return all_messages, existing_summary

    recent, new_summary = build_context_with_summary(all_messages, existing_summary)

    if new_summary and new_summary != existing_summary:
        db.update_conversation_summary(conversation_id, new_summary)

    return recent, new_summary


def save_turn(
    conversation_id: str,
    user_text: str,
    assistant_text: str,
    nlp_meta: dict,
) -> None:
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


def start_new_conversation(user_id: str) -> str:
    return db.create_conversation(user_id)