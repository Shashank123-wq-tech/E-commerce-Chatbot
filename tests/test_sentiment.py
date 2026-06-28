"""tests/test_sentiment.py"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_streamlit_cache(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "cache_resource", lambda **kw: (lambda f: f))


def test_analyze_sentiment_positive():
    with patch("src.sentiment._load_sentiment_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[
            {"label": "POSITIVE", "score": 0.94},
            {"label": "NEGATIVE", "score": 0.06},
        ]]
        mock_load.return_value = mock_pipe

        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("I love this product!")

    assert result.label == "POSITIVE"
    assert result.score == pytest.approx(0.94, abs=1e-3)
    assert result.is_positive is True
    assert result.is_negative is False


def test_empty_text_returns_neutral():
    with patch("src.sentiment._load_sentiment_pipeline"):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("  ")

    assert result.label == "NEUTRAL"
    assert result.score == 1.0