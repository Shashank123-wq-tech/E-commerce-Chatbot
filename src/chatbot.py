"""
chatbot.py — Orchestrates NLP pipeline + Groq LLM into a single interface.

Design decisions:
  • ChatBot is a thin coordinator — models are cached at MODULE level via
    @st.cache_resource in each sub-module, not inside this class.
  • process_stream()  → use with st.write_stream() for live typing effect
  • process()         → use for batch / testing without Streamlit
  • NLP analysis runs BEFORE the LLM call so the system prompt is enriched.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generator

from src.prompt_builder import build_from_text
from src.groq_client import stream_response, get_response
from src.ner import extract_entities, Entity
from src.sentiment import SentimentResult
from src.config import config
from src import memory


@dataclass
class Turn:
    """Represents one user-assistant exchange with its NLP metadata."""
    role: str               # "user" | "assistant"
    content: str
    intent: str = ""
    sentiment: str = ""
    entities: list[Entity] = field(default_factory=list)


class ChatBot:
    """
    Stateless wrapper — history is stored in st.session_state by app.py.
    Each call is self-contained; pass full history each time.
    """

    # ── Streaming (use in Streamlit) ──────────────────────────────────────────
    def process_stream(
        self,
        user_input: str,
        conversation_id: str,           # ← CHANGED: was `history: list[dict]`
    ) -> tuple[Generator[str, None, None], dict]:
        """
            (token_generator, nlp_meta)
        """
        # 1. NLP analysis (three cached models — no reload cost)
        system_prompt, intent_label, sentiment = build_from_text(user_input)
        entities = extract_entities(user_input)
        
        # 2 Get Context from DB - summary + recent messages (NEW)
        recent_messages, summary = memory.get_llm_context(conversation_id)
        
        # 3. Prepend summary into the system prompt if it exists
        if summary:
            system_prompt = (
                f"Context from earlier in this conversation: {summary}\n\n"
                f"{system_prompt}"
            )
        
        # 2. Build message history for Groq (trim to last N turns)
        groq_messages = recent_messages + [{"role": "user", "content": user_input}]
        
         
        # 3. Create streaming generator
        gen = stream_response(groq_messages, system_prompt)

        nlp_meta = {
            "intent":    intent_label,
            "sentiment": f"{sentiment.label} ({sentiment.score:.0%})",
            "entities":  [str(e) for e in entities],
        }
        return gen, nlp_meta

    # ── Non-streaming (for tests / CLI) ───────────────────────────────────────
    def process(self, user_input: str, conversation_id: str) -> tuple[str, dict]:
        """Returns (full_response_string, nlp_meta)."""
        system_prompt, intent_label, sentiment = build_from_text(user_input)
        entities = extract_entities(user_input)

        recent_messages, summary = memory.get_llm_context(conversation_id)
        if summary:
            system_prompt = f"Context from earlier: {summary}\n\n{system_prompt}"

        groq_messages = recent_messages + [{"role": "user", "content": user_input}] 
        response = get_response(groq_messages, system_prompt)

        nlp_meta = {
            "intent":    intent_label,
            "sentiment": f"{sentiment.label} ({sentiment.score:.0%})",
            "entities":  [str(e) for e in entities],
        }
        return response, nlp_meta
