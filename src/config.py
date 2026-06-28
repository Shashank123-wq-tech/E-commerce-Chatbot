"""
config.py — Central configuration for all models and settings.
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


@dataclass
class Config:
    # ── HuggingFace model repo IDs (set your username) ────────────────────────
    HF_USERNAME: str          = os.getenv("HF_USERNAME", "your-hf-username")
    INTENT_MODEL_ID: str      = field(init=False)
    NER_MODEL_ID: str         = field(init=False)
    SENTIMENT_MODEL_ID: str   = field(init=False)

    # ── Groq LLM settings ─────────────────────────────────────────────────────
    GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str     = "llama3-8b-8192"          # fast & cheap
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.7

    # ── Inference settings ────────────────────────────────────────────────────
    MAX_SEQ_LENGTH: int  = 256          # keep short → faster tokenization
    PIPELINE_DEVICE: int = field(default_factory=_pipeline_device)
    TORCH_DEVICE: str    = field(default_factory=_get_device)

    # ── Streamlit UI ──────────────────────────────────────────────────────────
    APP_TITLE: str       = "AI Powered Chatbot"
    APP_ICON: str        = "🤖"
    MAX_HISTORY: int     = 20           # keep last N turns in context

    def __post_init__(self):
        self.INTENT_MODEL_ID    = f"{self.HF_USERNAME}/intent-classifier"
        self.NER_MODEL_ID       = f"{self.HF_USERNAME}/ner-model"
        self.SENTIMENT_MODEL_ID = f"{self.HF_USERNAME}/sentiment-model"


config = Config()