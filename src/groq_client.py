from groq import Groq
import os
import streamlit as st
from typing import Generator


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


def generate_response(prompt: str) -> str:
    """Non-streaming — returns full response at once."""
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


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

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=full_messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=full_messages,
        stream=False,
    )
    return response.choices[0].message.content