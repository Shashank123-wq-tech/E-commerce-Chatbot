from groq import Groq
import os
import streamlit as st
from typing import Generator
from src.config import config


def _get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_client() -> Groq:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in secrets or environment.")
    return Groq(api_key=api_key)


# ── Fallback chain — tries models in order if one fails ────────────────────────
# Groq frequently deprecates/swaps free-tier models, so this makes the app
# resilient instead of breaking completely when one model goes away.
MODEL_FALLBACK_CHAIN = [
    config.GROQ_MODEL,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def _try_models(create_fn):
    """
    Tries each model in the fallback chain until one succeeds.
    create_fn(model_name) should perform the actual API call.
    """
    last_error = None
    tried = set()
    for model in MODEL_FALLBACK_CHAIN:
        if model in tried:
            continue
        tried.add(model)
        try:
            return create_fn(model)
        except Exception as e:
            last_error = e
            continue
    raise last_error


def generate_response(prompt: str) -> str:
    """Non-streaming — returns full response at once."""
    client = get_client()

    def _call(model):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content

    return _try_models(_call)


def stream_response(
    messages: list[dict],
    system_prompt: str = "",
) -> Generator[str, None, None]:
    """
    Streaming — yields tokens one by one.
    Use with st.write_stream() in app.py.
    """
    client = get_client()

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    last_error = None
    for model in MODEL_FALLBACK_CHAIN:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=full_messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return   # success — stop trying other models
        except Exception as e:
            last_error = e
            continue

    # All models failed
    yield f"⚠️ Sorry, the assistant is temporarily unavailable ({last_error})."


def get_response(
    messages: list[dict],
    system_prompt: str = "",
) -> str:
    """Non-streaming with full message history — used in chatbot.py."""
    client = get_client()

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    def _call(model):
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=False,
        )
        return response.choices[0].message.content

    return _try_models(_call)