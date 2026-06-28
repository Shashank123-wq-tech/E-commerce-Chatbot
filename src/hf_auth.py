"""
hf_auth.py — HuggingFace authentication helper.

Call setup_hf_auth() once at app startup (top of app.py).

• Public models  → no token needed, works out of the box
• Private models → needs HF_TOKEN in secrets
"""

import streamlit as st
from src.config import config


@st.cache_resource(show_spinner=False)
def setup_hf_auth() -> bool:
    """
    Logs into HuggingFace Hub if a token is provided.
    Safe to call even when no token is set (public models only).

    Returns True if authenticated, False if running without a token.
    """
    if not config.HF_TOKEN:
        # Public models don't need a token — from_pretrained works as-is
        return False

    try:
        from huggingface_hub import login
        login(token=config.HF_TOKEN, add_to_git_credential=False)
        print("✅ HuggingFace: authenticated successfully")
        return True
    except Exception as e:
        st.warning(f"⚠️ HuggingFace login failed: {e}. Only public models will work.")
        return False


def get_model_info() -> dict:
    """
    Returns model repo IDs for display in the sidebar.
    Useful for debugging / confirming which models are loaded.
    """
    return {
        "Intent Model":    config.INTENT_MODEL_ID,
        "NER Model":       config.NER_MODEL_ID,
        "Sentiment Model": config.SENTIMENT_MODEL_ID,
    }