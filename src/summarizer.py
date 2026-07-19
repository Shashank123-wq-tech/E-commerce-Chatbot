"""
summarizer.py — Conversation summarization to reduce token usage.

Strategy:
  • Keep the last N messages in FULL (recent context matters most)
  • Summarize everything OLDER than that into one short paragraph
  • Send (summary + last N messages) to Groq instead of full history
  • This keeps token usage flat even as conversations grow to 100+ turns
"""

from __future__ import annotations
import streamlit as st
from src.groq_client import get_client
from src.config import config

# How many recent messages to always keep in full detail
KEEP_RECENT_MESSAGES = 6

# Trigger summarization once conversation exceeds this many messages
SUMMARIZE_THRESHOLD = 10


def needs_summarization(message_count: int) -> bool:
    return message_count > SUMMARIZE_THRESHOLD


def summarize_messages(
    old_messages: list[dict],
    existing_summary: str | None = None,
) -> str:
    """
    Compresses a list of old messages (+ optional existing summary)
    into one short paragraph using Groq.

    Args:
        old_messages: [{"role": "user"/"assistant", "content": "..."}]
        existing_summary: previous summary to extend, if any

    Returns:
        Updated summary string.
    """
    if not old_messages:
        return existing_summary or ""

    client = get_client()

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_messages
    )

    system_prompt = (
        "You are a summarization assistant for a customer support chatbot. "
        "Summarize the conversation below into 3-5 short sentences. "
        "Capture: the customer's issue/intent, key details mentioned "
        "(order IDs, products, complaints), and how it was resolved or "
        "where it stands. Be factual and concise — no filler."
    )

    if existing_summary:
        system_prompt += f"\n\nExisting summary so far: {existing_summary}\nExtend it with the new messages below."

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": convo_text},
        ],
        max_tokens=200,
        temperature=0.3,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def build_context_with_summary(
    all_messages: list[dict],
    existing_summary: str | None,
) -> tuple[list[dict], str | None]:
    """
    Splits messages into (recent_messages_to_send, summary_to_prepend).

    Returns:
        (recent_messages, summary_text)
        • recent_messages → last KEEP_RECENT_MESSAGES, sent in full to Groq
        • summary_text    → summary of everything older, or None
    """
    if len(all_messages) <= KEEP_RECENT_MESSAGES:
        return all_messages, existing_summary

    recent  = all_messages[-KEEP_RECENT_MESSAGES:]
    to_fold = all_messages[:-KEEP_RECENT_MESSAGES]

    # Only re-summarize if there's new content to fold in
    if to_fold:
        new_summary = summarize_messages(to_fold, existing_summary)
        return recent, new_summary

    return recent, existing_summary