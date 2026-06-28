from groq import Groq
import os
import streamlit as st


def _get_api_key() -> str:
    """Reads from st.secrets first, then falls back to environment variable."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]   # ← square brackets, not ()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_client() -> Groq:
    """Cached Groq client — created once, reused forever."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in secrets or environment.")
    return Groq(api_key=api_key)


def generate_response(prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content