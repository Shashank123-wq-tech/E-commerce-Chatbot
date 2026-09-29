"""
analytics.py — Aggregates sentiment, intent, and entity data across
ALL conversations and ALL users for the analytics dashboard.

All functions are read-only queries against the existing messages/
conversations tables — no schema changes needed.
"""

from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
import streamlit as st
from sqlalchemy import text
from src.database import get_engine


@st.cache_data(ttl=300, show_spinner=False)  # cache 5 min — dashboard doesn't need live-live data
def get_intent_distribution(days: int = 30) -> dict[str, int]:
    """Returns {intent_name: count} for assistant messages in the last N days."""
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT intent, COUNT(*) as cnt
                FROM messages
                WHERE role = 'assistant'
                  AND intent IS NOT NULL
                  AND created_at >= :cutoff
                GROUP BY intent
                ORDER BY cnt DESC
            """),
            {"cutoff": cutoff}
        ).fetchall()

    return {r[0]: r[1] for r in rows}


@st.cache_data(ttl=300, show_spinner=False)
def get_sentiment_distribution(days: int = 30) -> dict[str, int]:
    """Returns {sentiment_label: count}, e.g. {'POSITIVE': 40, 'NEGATIVE': 12, 'NEUTRAL': 8}."""
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT sentiment, created_at
                FROM messages
                WHERE role = 'assistant'
                  AND sentiment IS NOT NULL
                  AND created_at >= :cutoff
            """),
            {"cutoff": cutoff}
        ).fetchall()

    counts = Counter()
    for r in rows:
        # sentiment stored like "POSITIVE (91%)" — extract just the label
        label = r[0].split(" ")[0].upper()
        counts[label] += 1

    return dict(counts)


@st.cache_data(ttl=300, show_spinner=False)
def get_sentiment_trend(days: int = 14) -> list[dict]:
    """
    Returns daily sentiment breakdown for a trend line chart:
    [{"date": "2026-07-01", "POSITIVE": 5, "NEGATIVE": 2, "NEUTRAL": 1}, ...]
    """
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT sentiment, created_at
                FROM messages
                WHERE role = 'assistant'
                  AND sentiment IS NOT NULL
                  AND created_at >= :cutoff
                ORDER BY created_at ASC
            """),
            {"cutoff": cutoff}
        ).fetchall()

    daily = {}
    for r in rows:
        label = r[0].split(" ")[0].upper()
        day = r[1].strftime("%Y-%m-%d")
        daily.setdefault(day, {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0})
        if label in daily[day]:
            daily[day][label] += 1

    return [{"date": d, **counts} for d, counts in sorted(daily.items())]


@st.cache_data(ttl=300, show_spinner=False)
def get_top_entities(days: int = 30, limit: int = 15) -> list[tuple[str, int]]:
    """Returns [(entity_type, count), ...] — most frequently detected entity types."""
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT entities
                FROM messages
                WHERE role = 'assistant'
                  AND entities IS NOT NULL
                  AND created_at >= :cutoff
            """),
            {"cutoff": cutoff}
        ).fetchall()

    type_counter = Counter()
    for r in rows:
        entities = r[0]
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except (json.JSONDecodeError, TypeError):
                entities = []
        for e in (entities or []):
            # entity strings look like: Entity('mumbai', DELIVERY_CITY, 0.96)
            try:
                ent_type = str(e).split(",")[1].strip()
                type_counter[ent_type] += 1
            except IndexError:
                continue

    return type_counter.most_common(limit)


@st.cache_data(ttl=300, show_spinner=False)
def get_overview_stats(days: int = 30) -> dict:
    """Returns high-level KPI numbers for the top of the dashboard."""
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.begin() as conn:
        total_messages = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE created_at >= :cutoff"),
            {"cutoff": cutoff}
        ).scalar()

        total_conversations = conn.execute(
            text("SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE created_at >= :cutoff"),
            {"cutoff": cutoff}
        ).scalar()

        total_users = conn.execute(
            text("""
                SELECT COUNT(DISTINCT c.user_id)
                FROM conversations c
                JOIN messages m ON m.conversation_id = c.id
                WHERE m.created_at >= :cutoff
            """),
            {"cutoff": cutoff}
        ).scalar()

    sentiment_dist = get_sentiment_distribution(days)
    total_sentiment = sum(sentiment_dist.values()) or 1
    positive_pct = round(sentiment_dist.get("POSITIVE", 0) / total_sentiment * 100, 1)
    negative_pct = round(sentiment_dist.get("NEGATIVE", 0) / total_sentiment * 100, 1)

    return {
        "total_messages": total_messages or 0,
        "total_conversations": total_conversations or 0,
        "total_users": total_users or 0,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
    }
    
    
# ══════════════════════════════════════════════════════════════════════════════
# ADD these functions to the END of your existing src/analytics.py
# (keep everything already in that file — this is additive)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_all_customers() -> list[dict]:
    """
    Returns list of all customers who have chatted, for the dropdown selector:
    [{"user_id": "...", "email": "...", "name": "...", "message_count": 12}, ...]
    """
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT u.id, u.email, u.name, COUNT(m.id) as msg_count
                FROM users u
                JOIN conversations c ON c.user_id = u.id
                JOIN messages m ON m.conversation_id = c.id
                GROUP BY u.id, u.email, u.name
                ORDER BY msg_count DESC
            """)
        ).fetchall()

    return [
        {
            "user_id": str(r[0]),
            "email": r[1],
            "name": r[2] or r[1].split("@")[0],
            "message_count": r[3],
        }
        for r in rows
    ]


@st.cache_data(ttl=120, show_spinner=False)
def get_customer_messages(user_id: str, limit: int = 200) -> list[dict]:
    """
    Returns every message (with NLP metadata) for ONE specific customer,
    across all their conversations, newest first.
    """
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT m.role, m.content, m.intent, m.sentiment, m.entities, m.created_at
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = :uid
                ORDER BY m.created_at DESC
                LIMIT :limit
            """),
            {"uid": user_id, "limit": limit}
        ).fetchall()

    result = []
    for r in rows:
        entities = r[4]
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except (json.JSONDecodeError, TypeError):
                entities = []
        result.append({
            "role": r[0],
            "content": r[1],
            "intent": r[2],
            "sentiment": r[3],
            "entities": entities,
            "created_at": r[5],
        })
    return result


@st.cache_data(ttl=120, show_spinner=False)
def get_customer_summary_stats(user_id: str) -> dict:
    """
    Returns high-level stats for one customer:
    total messages, dominant sentiment, dominant intent, first/last seen.
    """
    messages = get_customer_messages(user_id, limit=1000)
    assistant_msgs = [m for m in messages if m["role"] == "assistant" and m["intent"]]

    if not assistant_msgs:
        return {
            "total_messages": len(messages),
            "dominant_intent": "—",
            "dominant_sentiment": "—",
            "sentiment_score_avg": 0,
            "first_seen": None,
            "last_seen": None,
        }

    intent_counts = Counter(m["intent"] for m in assistant_msgs)
    sentiment_labels = [m["sentiment"].split(" ")[0].upper() for m in assistant_msgs if m["sentiment"]]
    sentiment_counts = Counter(sentiment_labels)

    dates = [m["created_at"] for m in messages if m["created_at"]]

    return {
        "total_messages": len(messages),
        "dominant_intent": intent_counts.most_common(1)[0][0] if intent_counts else "—",
        "dominant_sentiment": sentiment_counts.most_common(1)[0][0] if sentiment_counts else "—",
        "sentiment_breakdown": dict(sentiment_counts),
        "intent_breakdown": dict(intent_counts),
        "first_seen": min(dates) if dates else None,
        "last_seen": max(dates) if dates else None,
    }    