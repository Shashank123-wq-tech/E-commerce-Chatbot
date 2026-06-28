"""
utils.py — Shared helpers used across the project.
"""

from __future__ import annotations
import re
import time
from functools import wraps
from typing import Callable, Any


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Light normalisation before sending to NLP models."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    text = re.sub(r"http\S+", "[URL]", text)   # mask URLs
    return text


def truncate(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Truncate long text for display in UI tooltips / logs."""
    return text[:max_chars] + suffix if len(text) > max_chars else text


# ── Timing decorator ───────────────────────────────────────────────────────────
def timeit(func: Callable) -> Callable:
    """Decorator that prints execution time — useful for profiling."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[timeit] {func.__name__} took {elapsed:.1f} ms")
        return result
    return wrapper


# ── Streamlit session helpers ──────────────────────────────────────────────────
def init_session_defaults(st_session: dict, defaults: dict) -> None:
    """
    Initialise session_state keys only if they don't already exist.
    Call once at the top of app.py.

    Usage:
        init_session_defaults(st.session_state, {
            "messages": [],
            "nlp_meta": {},
        })
    """
    for key, value in defaults.items():
        if key not in st_session:
            st_session[key] = value


# ── Chat history helpers ───────────────────────────────────────────────────────
def to_groq_history(messages: list[dict]) -> list[dict]:
    """
    Convert Streamlit-style chat history to the format Groq expects.
    Filters out any keys beyond 'role' and 'content'.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def format_entities_md(entities: list[str]) -> str:
    """Format entity list as a Markdown bullet list for sidebar display."""
    if not entities:
        return "_None detected_"
    return "\n".join(f"- `{e}`" for e in entities)