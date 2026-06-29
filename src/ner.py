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


@dataclass
class Entity:
    text: str
    label: str
    score: float
    start: int
    end: int

    def __repr__(self) -> str:
        return f"Entity({self.text!r}, {self.label}, {self.score:.2f})"


@st.cache_resource(show_spinner="⚙️ Loading NER model...")
def _load_ner_pipeline():
    from src.config import config

    tokenizer = AutoTokenizer.from_pretrained(
        config.NER_MODEL_ID,
        token=config.HF_TOKEN or None
    )
    model = AutoModelForTokenClassification.from_pretrained(
        config.NER_MODEL_ID,
        token=  config.HF_TOKEN or None,
        torch_dtype=torch.float32,
    )
    model.eval()
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
        truncation=True,
        max_length=256,
    )


def extract_entities(text: str) -> list[Entity]:
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
    result: dict[str, list[str]] = {}
    for ent in extract_entities(text):
        result.setdefault(ent.label, []).append(ent.text)
    return result