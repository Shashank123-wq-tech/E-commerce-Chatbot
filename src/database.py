"""
database.py — PostgreSQL connectivity layer.

Schema:
  users          → one row per browser session / user
  conversations  → one row per chat session (belongs to a user)
  messages       → every user + assistant message (belongs to a conversation)

Uses SQLAlchemy Core (lightweight, no ORM overhead) + connection pooling
via st.cache_resource so the engine is created once per app lifecycle.
"""

from __future__ import annotations
import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, text, Table, Column, MetaData,
    String, Integer, Text, TIMESTAMP, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
import uuid


# ── Engine (cached — created once) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    from src.config import config
    if not config.DATABASE_URL:
        raise ValueError("DATABASE_URL not set in secrets/.env")
    return create_engine(config.DATABASE_URL, pool_pre_ping=True, pool_size=5)


metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("session_key", String, unique=True, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)

conversations = Table(
    "conversations", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), nullable=False),
    Column("summary", Text, nullable=True),          # rolling summary of old messages
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column("updated_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)

messages = Table(
    "messages", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("conversation_id", UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False),
    Column("role", String, nullable=False),           # "user" | "assistant"
    Column("content", Text, nullable=False),
    Column("intent", String, nullable=True),
    Column("sentiment", String, nullable=True),
    Column("entities", JSON, nullable=True),
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)


def init_db() -> None:
    """Create tables if they don't exist. Call once at app startup."""
    engine = get_engine()
    metadata.create_all(engine)


# ── User helpers ────────────────────────────────────────────────────────────────
def get_or_create_user(session_key: str) -> str:
    """Returns user_id (str). Creates a user row if not found."""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM users WHERE session_key = :sk"),
            {"sk": session_key}
        ).fetchone()
        if row:
            return str(row[0])

        new_id = uuid.uuid4()
        conn.execute(
            text("INSERT INTO users (id, session_key) VALUES (:id, :sk)"),
            {"id": new_id, "sk": session_key}
        )
        return str(new_id)


# ── Conversation helpers ─────────────────────────────────────────────────────────
def create_conversation(user_id: str) -> str:
    engine = get_engine()
    new_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (id, user_id) VALUES (:id, :uid)"),
            {"id": new_id, "uid": user_id}
        )
    return str(new_id)


def get_latest_conversation(user_id: str) -> str | None:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT id FROM conversations
                WHERE user_id = :uid
                ORDER BY updated_at DESC
                LIMIT 1
            """),
            {"uid": user_id}
        ).fetchone()
        return str(row[0]) if row else None


def update_conversation_summary(conversation_id: str, summary: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE conversations
                SET summary = :summary, updated_at = now()
                WHERE id = :cid
            """),
            {"summary": summary, "cid": conversation_id}
        )


def get_conversation_summary(conversation_id: str) -> str | None:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT summary FROM conversations WHERE id = :cid"),
            {"cid": conversation_id}
        ).fetchone()
        return row[0] if row and row[0] else None


# ── Message helpers ───────────────────────────────────────────────────────────────
def save_message(
    conversation_id: str,
    role: str,
    content: str,
    intent: str = None,
    sentiment: str = None,
    entities: list = None,
) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages
                    (id, conversation_id, role, content, intent, sentiment, entities)
                VALUES
                    (:id, :cid, :role, :content, :intent, :sentiment, :entities)
            """),
            {
                "id": uuid.uuid4(),
                "cid": conversation_id,
                "role": role,
                "content": content,
                "intent": intent,
                "sentiment": sentiment,
                "entities": entities or [],
            }
        )
        conn.execute(
            text("UPDATE conversations SET updated_at = now() WHERE id = :cid"),
            {"cid": conversation_id}
        )


def get_messages(conversation_id: str, limit: int = 100) -> list[dict]:
    """Returns messages in chronological order (oldest first)."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, intent, sentiment, entities, created_at
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at ASC
                LIMIT :limit
            """),
            {"cid": conversation_id, "limit": limit}
        ).fetchall()

    return [
        {
            "role": r[0],
            "content": r[1],
            "intent": r[2],
            "sentiment": r[3],
            "entities": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def count_messages(conversation_id: str) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE conversation_id = :cid"),
            {"cid": conversation_id}
        ).fetchone()
        return row[0] if row else 0


def delete_conversation_messages(conversation_id: str) -> None:
    """Used by 'Clear chat' — wipes messages but keeps the conversation row."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM messages WHERE conversation_id = :cid"),
            {"cid": conversation_id}
        )
        conn.execute(
            text("UPDATE conversations SET summary = NULL WHERE id = :cid"),
            {"cid": conversation_id}
        )