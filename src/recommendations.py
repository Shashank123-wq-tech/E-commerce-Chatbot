"""
recommendations.py — Generates business recommendations using Groq,
based on the aggregated intent/sentiment/entity data from analytics.py.

This is what makes the dashboard "smart" rather than just showing charts —
it turns numbers into actionable suggestions for the business.
"""

from __future__ import annotations
import streamlit as st
from src.groq_client import get_client
from src.config import config


@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 min — recommendations don't need to regenerate every reload
def generate_recommendations(
    intent_distribution: dict[str, int],
    sentiment_distribution: dict[str, int],
    top_entities: list[tuple[str, int]],
) -> str:
    """
    Returns a markdown-formatted string with 3-5 actionable business
    recommendations based on the aggregated chatbot data.
    """
    if not intent_distribution and not sentiment_distribution:
        return "_Not enough conversation data yet to generate recommendations._"

    intent_summary = ", ".join(f"{k}: {v}" for k, v in list(intent_distribution.items())[:10])
    sentiment_summary = ", ".join(f"{k}: {v}" for k, v in sentiment_distribution.items())
    entity_summary = ", ".join(f"{k}: {v}" for k, v in top_entities[:10])

    prompt = f"""You are a business analyst reviewing customer support chatbot data
for an e-commerce company.

DATA SUMMARY:
- Customer intents (what they're asking about): {intent_summary}
- Overall sentiment distribution: {sentiment_summary}
- Most mentioned entity types: {entity_summary}

Based on this data, write 3-5 short, specific, actionable business
recommendations. Focus on:
- Which operational areas need attention (e.g. if "delivery_issue" is high, recommend logistics review)
- Whether sentiment trends suggest a customer experience problem
- Any patterns in entities that suggest a specific product/region issue

Format as a markdown bullet list. Each bullet should be 1-2 sentences,
specific and actionable — not generic advice. Start directly with the
first bullet point, no preamble."""

    client = get_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.4,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# ADD this function to the END of your existing src/recommendations.py
# (keep everything already in that file — this is additive)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def generate_customer_insight(
    customer_name: str,
    intent_breakdown: dict[str, int],
    sentiment_breakdown: dict[str, int],
    recent_messages: list[str],
) -> str:
    """
    Returns a short, specific insight about ONE customer — used for
    customer support teams to quickly understand a single user's situation.
    """
    if not intent_breakdown and not sentiment_breakdown:
        return "_Not enough messages yet from this customer to generate an insight._"

    intent_summary = ", ".join(f"{k}: {v}" for k, v in intent_breakdown.items())
    sentiment_summary = ", ".join(f"{k}: {v}" for k, v in sentiment_breakdown.items())
    recent_summary = "\n".join(f"- {m}" for m in recent_messages[:5])

    prompt = f"""You are a customer support analyst. Here is data about ONE
specific customer named {customer_name}:

Intent history: {intent_summary}
Sentiment history: {sentiment_summary}
Recent messages:
{recent_summary}

Write a short 2-3 sentence insight for the support team about this specific
customer: what they likely need, their emotional state, and one concrete
next action the support team should take. Be specific and direct — no
generic advice. Start directly with the insight, no preamble."""

    client = get_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4,
        stream=False,
    )
    return response.choices[0].message.content.strip()

