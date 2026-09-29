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