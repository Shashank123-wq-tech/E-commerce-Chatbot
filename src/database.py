"""
database.py — PostgreSQL connectivity layer.

MODIFIED for Google Login: password_hash is now nullable since Google
handles authentication — we just need email + name from the OIDC token.
"""

from __future__ import annotations
import json
import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, text, Table, Column, MetaData,
    String, Integer, Text, TIMESTAMP, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
import uuid


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
    Column("email", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=True),   # ← nullable now (Google users don't have one)
    Column("name", String, nullable=True),
    Column("picture", String, nullable=True),          # ← NEW: Google profile photo URL
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)

conversations = Table(
    "conversations", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), nullable=False),
    Column("summary", Text, nullable=True),
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column("updated_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)

messages = Table(
    "messages", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("conversation_id", UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("intent", String, nullable=True),
    Column("sentiment", String, nullable=True),
    Column("entities", JSON, nullable=True),
    Column("created_at", TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)),
)


def init_db() -> None:
    engine = get_engine()
    metadata.create_all(engine)


# ══════════════════════════════════════════════════════════════════════════════
# Google-auth user helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_user_by_email(email: str, name: str = None, picture: str = None) -> str:
    """
    Returns user_id. Creates the user on their first Google login,
    otherwise returns their existing id (and refreshes name/picture
    in case they changed on Google's side).
    """
    engine = get_engine()
    email = email.strip().lower()

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()

        if row:
            # Keep name/picture fresh on every login
            conn.execute(
                text("UPDATE users SET name = :name, picture = :picture WHERE id = :id"),
                {"name": name, "picture": picture, "id": row[0]}
            )
            return str(row[0])

        new_id = uuid.uuid4()
        conn.execute(
            text("""
                INSERT INTO users (id, email, name, picture)
                VALUES (:id, :email, :name, :picture)
            """),
            {"id": new_id, "email": email, "name": name, "picture": picture}
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
                "entities": json.dumps(entities or []),
            }
        )
        conn.execute(
            text("UPDATE conversations SET updated_at = now() WHERE id = :cid"),
            {"cid": conversation_id}
        )


def get_messages(conversation_id: str, limit: int = 100) -> list[dict]:
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


def count_messages(conversation_id: str) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE conversation_id = :cid"),
            {"cid": conversation_id}
        ).fetchone()
        return row[0] if row else 0


def delete_conversation_messages(conversation_id: str) -> None:
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