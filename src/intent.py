"""
intent.py — Intent classification from your HuggingFace model.

Key optimisations:
  • @st.cache_resource  → model loads ONCE per server session, never on re-run
  • model.eval()        → disables dropout / batch-norm training behaviour
  • torch.no_grad()     → skips gradient tracking → ~30 % less memory & faster
  • truncation + max_length → avoids OOM on long inputs
"""

import json
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
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
    
@st.cache_resource(show_spinner="⚙️ Loading id2intent mapping...")
def _load_id2intent() -> dict:
    from src.config import config

    # ── Try 1: load from id2intent.json ───────────────────────────────────────
    try:
        file_path = hf_hub_download(
            repo_id=config.INTENT_MODEL_ID,
            filename="id2intent.json",
            token=config.HF_TOKEN or None,
        )
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Handle both {"0": "greeting"} and {"LABEL_0": "greeting"} formats
        mapping = {}
        for k, v in raw.items():
            key = str(k).replace("LABEL_", "")   # strip LABEL_ if present
            mapping[f"LABEL_{key}"] = v           # normalise to LABEL_X format
        return mapping

    except Exception as e:
        pass

    # ── Try 2: load from intent2id.json and reverse it ────────────────────────
    try:
        file_path = hf_hub_download(
            repo_id=config.INTENT_MODEL_ID,
            filename="intent2id.json",
            token=config.HF_TOKEN or None,
        )
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # intent2id is {"greeting": 0} → reverse to {0: "greeting"}
        mapping = {}
        for intent_name, label_id in raw.items():
            mapping[f"LABEL_{label_id}"] = intent_name
        return mapping

    except Exception as e:
        pass

    # ── Try 3: read directly from model config ────────────────────────────────
    try:
        clf = _load_intent_pipeline()
        id2label = clf.model.config.id2label
        return {f"LABEL_{k}": v for k, v in id2label.items()}
    except Exception:
        return {}    


def classify_intent(text: str) -> dict:
    if not text.strip():
        return {}

    clf      = _load_intent_pipeline()
    id2intent = _load_id2intent()

    with torch.no_grad():
        results = clf(text)

    scores = {}
    for r in results[0]:
        raw_label  = r["label"]                              # e.g. "LABEL_19"
        real_label = id2intent.get(raw_label, raw_label)    # e.g. "delivery_issue"
        scores[real_label] = round(r["score"], 4)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def top_intent(text: str) -> tuple:
    scores = classify_intent(text)
    if not scores:
        return "unknown", 0.0
    label, conf = next(iter(scores.items()))
    return label, conf


