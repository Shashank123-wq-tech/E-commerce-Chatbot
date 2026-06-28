"""
tests/test_intent.py

Run:  pytest tests/test_intent.py -v
Note: Uses the non-Streamlit path — models are loaded directly (no cache decorator).
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Patch Streamlit cache before importing the module ─────────────────────────
@pytest.fixture(autouse=True)
def mock_streamlit_cache(monkeypatch):
    """Replace @st.cache_resource with a passthrough so tests work without Streamlit."""
    import streamlit as st
    monkeypatch.setattr(st, "cache_resource", lambda **kw: (lambda f: f))


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_classify_intent_returns_dict():
    with patch("src.intent._load_intent_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "greeting", "score": 0.92},
            {"label": "unknown",  "score": 0.08},
        ]]
        mock_load.return_value = mock_pipe

        from src.intent import classify_intent
        result = classify_intent("Hello there!")

    assert isinstance(result, dict)
    assert "greeting" in result
    assert result["greeting"] == pytest.approx(0.92, abs=1e-3)


def test_top_intent_returns_best_label():
    with patch("src.intent._load_intent_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "book_flight", "score": 0.88},
            {"label": "greeting",   "score": 0.12},
        ]]
        mock_load.return_value = mock_pipe

        from src.intent import top_intent
        label, conf = top_intent("I want to fly to Delhi tomorrow")

    assert label == "book_flight"
    assert conf == pytest.approx(0.88, abs=1e-3)


def test_empty_text_returns_empty():
    with patch("src.intent._load_intent_pipeline") as mock_load:
        from src.intent import classify_intent
        result = classify_intent("   ")

    assert result == {}
    mock_load.assert_not_called()  