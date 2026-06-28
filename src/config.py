"""
Config.py — Central Configuration for all models and settings.
Update HF_USERNAME and model repo names to match your HuggingFace uploads.
"""

import os
import torch
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ── Auto-detect best available device ─────────────────────────────────────────
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():   # Apple Silicon
        return "mps"
    return "cpu"


def _pipeline_device() -> int:
    """transformers pipeline expects -1 for CPU, 0 for first GPU."""
    dev = _get_device()
    return 0 if dev in ("cuda", "mps") else -1


def _secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
        return os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)


@dataclass
class Config:
    # HuggingFace
    HF_USERNAME:        str = field(default_factory=lambda: _secret("HF_USERNAME", "dixitshashank937"))
    HF_TOKEN:           str = field(default_factory=lambda: _secret("HF_TOKEN", ""))

    # Read model IDs directly from secrets so you can fix them
    # without changing code — just update Streamlit secrets
    INTENT_MODEL_ID:    str = field(default_factory=lambda: _secret(
                                "INTENT_MODEL_ID", "dixitshashank937/intent-classifier"))
    NER_MODEL_ID:       str = field(default_factory=lambda: _secret(
                                "NER_MODEL_ID", "dixitshashank937/ner-model"))
    SENTIMENT_MODEL_ID: str = field(default_factory=lambda: _secret(
                                "SENTIMENT_MODEL_ID", "dixitshashank937/sentiment-model"))

    # Groq
    GROQ_API_KEY:     str   = field(default_factory=lambda: _secret("GROQ_API_KEY", ""))
    GROQ_MODEL:       str   = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS:  int   = 1024
    GROQ_TEMPERATURE: float = 0.7

    # Inference
    MAX_SEQ_LENGTH:  int = 256
    PIPELINE_DEVICE: int = field(default_factory=_pipeline_device)
    TORCH_DEVICE:    str = field(default_factory=_get_device)

    # UI
    APP_TITLE:   str = "AI Powered Chatbot"
    APP_ICON:    str = "🤖"
    MAX_HISTORY: int = 20