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
        token=config.HF_TOKEN or None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.SENTIMENT_MODEL_ID,
        token=config.HF_TOKEN or None,
        torch_dtype=torch.float32,
    )
    model.eval()
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=None,
        truncation=True,
        max_length=256,
    )


def analyze_sentiment(text: str) -> SentimentResult:
    if not text.strip():
        return SentimentResult("NEUTRAL", 1.0, {"NEUTRAL": 1.0})
    clf = _load_sentiment_pipeline()
    with torch.no_grad():
        raw = clf(text)
    all_scores = {r["label"]: round(r["score"], 4) for r in raw[0]}
    best = max(raw[0], key=lambda x: x["score"])
    return SentimentResult(
        label=best["label"],
        score=round(best["score"], 4),
        all_scores=all_scores,
    )