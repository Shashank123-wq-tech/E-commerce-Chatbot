"""
ner.py — Named Entity Recognition from your HuggingFace model.

Key optimisations:
  • @st.cache_resource        → single load per server lifecycle
  • aggregation_strategy="simple" → merges B-/I- tokens automatically
  • torch.no_grad()           → no gradient overhead
  • Returns clean dataclass objects instead of raw dicts
"""

from __future__ import annotations

import streamlit as st
import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from src.config import config


@dataclass
class Entity:
    text: str
    label: str
    score: float
    start: int
    end: int

    def __repr__(self) -> str:
        return f"Entity({self.text!r}, {self.label}, {self.score:.2f})"


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading NER model…")
def _load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("dixitshashank937/ner-model1")
    model = AutoModelForTokenClassification.from_pretrained(
        config.NER_MODEL_ID,
        torch_dtype=torch.float16 if config.TORCH_DEVICE == "cuda" else torch.float32,
    )
    model.eval()
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",   # auto-merge sub-word tokens
        device=config.PIPELINE_DEVICE,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def extract_entities(text: str) -> list[Entity]:
    """
    Returns a list of Entity objects found in `text`.
    Example: [Entity('Delhi', 'GPE', 0.98, 10, 15), ...]
    """
    if not text.strip():
        return []

    ner = _load_ner_pipeline()

    with torch.no_grad():
        raw = ner(text)

    return [
        Entity(
            text=span["word"].strip(),
            label=span["entity_group"],
            score=round(span["score"], 4),
            start=span["start"],
            end=span["end"],
        )
        for span in raw
    ]


def entities_by_type(text: str) -> dict[str, list[str]]:
    """
    Groups entity texts by their label type.
    Example: {"GPE": ["Delhi", "Mumbai"], "PERSON": ["Rahul"]}
    """
    result: dict[str, list[str]] = {}
    for ent in extract_entities(text):
        result.setdefault(ent.label, []).append(ent.text)
    return result