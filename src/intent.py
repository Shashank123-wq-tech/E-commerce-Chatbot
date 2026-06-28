"""
intent.py — Intent classification from your HuggingFace model.

Key optimisations:
  • @st.cache_resource  → model loads ONCE per server session, never on re-run
  • model.eval()        → disables dropout / batch-norm training behaviour
  • torch.no_grad()     → skips gradient tracking → ~30 % less memory & faster
  • truncation + max_length → avoids OOM on long inputs
"""

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from src.config import config


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading Intent model…")
def _load_intent_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("dixitshashank937/intent-classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.INTENT_MODEL_ID,
        torch_dtype=torch.float16 if config.TORCH_DEVICE == "cuda" else torch.float32,
    )
    model.eval()                        # inference mode
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=config.PIPELINE_DEVICE,
        top_k=None,                     # return ALL label scores
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def classify_intent(text: str) -> dict[str, float]:
    """
    Returns a dict of {intent_label: confidence_score}, sorted descending.
    Example: {"book_flight": 0.91, "cancel_order": 0.05, ...}
    """
    if not text.strip():
        return {}

    clf = _load_intent_pipeline()

    with torch.no_grad():
        results = clf(text)             # list of [{"label": ..., "score": ...}]

    scores = {r["label"]: round(r["score"], 4) for r in results[0]}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def top_intent(text: str) -> tuple[str, float]:
    """Convenience wrapper → returns (best_label, confidence)."""
    scores = classify_intent(text)
    if not scores:
        return "unknown", 0.0
    label, conf = next(iter(scores.items()))
    return label, conf