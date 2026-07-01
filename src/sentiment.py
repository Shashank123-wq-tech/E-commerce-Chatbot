"""
sentiment.py — Sentiment analysis from your HuggingFace model.

Key optimisations:
  • @st.cache_resource  → load once, reuse forever
  • model.eval()        → disables dropout
  • torch.no_grad()     → skips backward graph
"""

from __future__ import annotations
import streamlit as st
import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


FALLBACK_SENTIMENT_MAP = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}



@dataclass
class SentimentResult:
    label: str
    score: float
    all_scores: dict[str, float]

    @property
    def is_positive(self) -> bool:
        return self.label.upper() == "POSITIVE"

    @property
    def is_negative(self) -> bool:
        return self.label.upper() == "NEGATIVE"

    def __repr__(self) -> str:
        return f"Sentiment({self.label}, {self.score:.2f})"


@st.cache_resource(show_spinner="⚙️ Loading Sentiment model...")
def _load_sentiment_pipeline():
    from src.config import config

    tokenizer = AutoTokenizer.from_pretrained(
        config.SENTIMENT_MODEL_ID,
        token=config.HF_TOKEN or None,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.SENTIMENT_MODEL_ID,
        token=config.HF_TOKEN or None,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    # ── Read label mapping directly from model config ─────────────────────────
    id2label = model.config.id2label
    # If model config has real names use them, else use fallback
    # id2label looks like: {0: "LABEL_0"} or {0: "negative"}

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=None,
        truncation=True,
        max_length=256,
    )
    return clf, id2label

def _resolve_label(raw_label: str, id2label: dict) -> str:
    """
    Converts LABEL_0 → real name using id2label from model config.
    Falls back to FALLBACK_SENTIMENT_MAP if model has no real names.
    """
    label_id = int(raw_label.replace("LABEL_", ""))
    model_label = id2label.get(label_id, raw_label)

    # If model still returns LABEL_X, use our fallback map
    if "LABEL_" in str(model_label).upper():
        return FALLBACK_SENTIMENT_MAP.get(label_id, raw_label).upper()

    return str(model_label).upper()

def analyze_sentiment(text: str) -> SentimentResult:
    if not text.strip():
        return SentimentResult("NEUTRAL", 1.0, {"NEUTRAL": 1.0})

    clf, id2label = _load_sentiment_pipeline()

    with torch.no_grad():
        raw = clf(text)

    all_scores = {
        _resolve_label(r["label"], id2label): round(r["score"], 4)
        for r in raw[0]
    }
    best = max(raw[0], key=lambda x: x["score"])

    return SentimentResult(
        label=_resolve_label(best["label"], id2label),
        score=round(best["score"], 4),
        all_scores=all_scores,
    )