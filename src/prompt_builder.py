"""
prompt_builder.py — Converts NLP analysis results into a dynamic system prompt.

The richer context you give the LLM, the more targeted its response.
"""

from __future__ import annotations
from src.intent import top_intent
from src.ner import entities_by_type
from src.sentiment import SentimentResult


# ── Tone mapping ───────────────────────────────────────────────────────────────
_TONE_MAP: dict[str, str] = {
    "POSITIVE": "The user seems happy or satisfied. Keep the tone warm and enthusiastic.",
    "NEGATIVE": "The user seems frustrated or unhappy. Be extra empathetic and solution-focused.",
    "NEUTRAL":  "The user is neutral. Be clear, concise, and professional.",
}

# ── Intent-specific instructions ──────────────────────────────────────────────
_INTENT_GUIDANCE: dict[str, str] = {
    "greeting":         "Greet the user warmly and ask how you can help.",
    "book_flight":      "Help the user book a flight. Ask for origin, destination, and travel dates if missing.",
    "cancel_order":     "Assist with order cancellation. Empathize and explain the process clearly.",
    "track_order":      "Help the user track their order. Ask for order ID if not mentioned.",
    "product_inquiry":  "Answer product questions accurately. Highlight key features and benefits.",
    "complaint":        "Acknowledge the issue, apologize sincerely, and provide clear resolution steps.",
    "faq":              "Provide a direct, helpful answer to the frequently asked question.",
    "unknown":          "The user's intent is unclear. Ask a clarifying question.",
}


def build_system_prompt(
    intent_label: str,
    intent_conf: float,
    sentiment: SentimentResult,
    entities: dict[str, list[str]],
) -> str:
    """
    Constructs a system prompt using NLP signals.

    Args:
        intent_label : Top predicted intent label
        intent_conf  : Confidence of that intent (0-1)
        sentiment    : SentimentResult object
        entities     : dict from entities_by_type(), e.g. {"GPE": ["Delhi"]}

    Returns:
        A formatted system prompt string.
    """
    # Base persona
    prompt_parts = [
        "You are a helpful, friendly AI assistant.",
        "Always respond in the same language the user uses.",
    ]

    # Intent guidance
    guidance = _INTENT_GUIDANCE.get(intent_label, _INTENT_GUIDANCE["unknown"])
    if intent_conf > 0.6:
        prompt_parts.append(f"User intent: **{intent_label}** ({intent_conf:.0%} confidence). {guidance}")
    else:
        prompt_parts.append("The user's intent is ambiguous. Ask a clarifying question before proceeding.")

    # Sentiment tone
    tone = _TONE_MAP.get(sentiment.label.upper(), _TONE_MAP["NEUTRAL"])
    prompt_parts.append(f"Tone guidance: {tone}")

    # Inject detected entities as context
    if entities:
        entity_str = "; ".join(
            f"{label}: {', '.join(vals)}" for label, vals in entities.items()
        )
        prompt_parts.append(f"Detected entities in the message: {entity_str}.")
        prompt_parts.append("Use these entities naturally in your response where relevant.")

    # Safety guardrail
    prompt_parts.append(
        "Never make up facts. If you are unsure, say so honestly and offer to help further."
    )

    return "\n".join(prompt_parts)


def build_from_text(user_text: str) -> tuple[str, str, SentimentResult]:
    """
    One-shot helper: runs all three NLP models and returns
    (system_prompt, intent_label, sentiment).

    Useful in chatbot.py to avoid importing each module separately.
    """
    from src.sentiment import analyze_sentiment

    intent_label, intent_conf = top_intent(user_text)
    sentiment                 = analyze_sentiment(user_text)
    entities                  = entities_by_type(user_text)

    system_prompt = build_system_prompt(intent_label, intent_conf, sentiment, entities)
    return system_prompt, intent_label, sentiment