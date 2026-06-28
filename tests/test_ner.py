"""tests/test_ner.py"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_streamlit_cache(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "cache_resource", lambda **kw: (lambda f: f))


def test_extract_entities_returns_list():
    with patch("src.ner._load_ner_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"word": "Delhi",  "entity_group": "GPE",    "score": 0.99, "start": 20, "end": 25},
            {"word": "Rahul",  "entity_group": "PERSON", "score": 0.97, "start": 0,  "end": 5},
        ]
        mock_load.return_value = mock_pipe

        from src.ner import extract_entities
        entities = extract_entities("Rahul wants to go to Delhi")

    assert len(entities) == 2
    labels = {e.label for e in entities}
    assert "GPE" in labels
    assert "PERSON" in labels


def test_entities_by_type_groups_correctly():
    with patch("src.ner._load_ner_pipeline") as mock_load:
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"word": "Mumbai", "entity_group": "GPE",    "score": 0.98, "start": 0,  "end": 6},
            {"word": "Delhi",  "entity_group": "GPE",    "score": 0.96, "start": 10, "end": 15},
            {"word": "Anita",  "entity_group": "PERSON", "score": 0.95, "start": 20, "end": 25},
        ]
        mock_load.return_value = mock_pipe

        from src.ner import entities_by_type
        grouped = entities_by_type("Mumbai and Delhi, Anita")

    assert "GPE" in grouped
    assert len(grouped["GPE"]) == 2
    assert "PERSON" in grouped