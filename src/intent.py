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


@st.cache_resource(show_spinner="⚙️ Loading Intent model...")
def _load_intent_pipeline():
    from src.config import config

    tokenizer = AutoTokenizer.from_pretrained(
        config.INTENT_MODEL_ID,
        token=config.HF_TOKEN or None
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.INTENT_MODEL_ID,
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


def classify_intent(text: str) -> dict:
    if not text.strip():
        return {}
    clf = _load_intent_pipeline()
    with torch.no_grad():
        results = clf(text)
    scores = {r["label"]: round(r["score"], 4) for r in results[0]}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def top_intent(text: str) -> tuple:
    scores = classify_intent(text)
    if not scores:
        return "unknown", 0.0
    label, conf = next(iter(scores.items()))
    return label, conf