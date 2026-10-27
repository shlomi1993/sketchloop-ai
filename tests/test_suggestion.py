"""
Tests of the prompt suggesters with a fake captioner, so no model is loaded.
"""

import pytest

from pathlib import Path

from sketchloop.suggestion import (NEUTRAL_SUBJECT, STYLE_TEMPLATES, FlorenceSuggester, SuggestionError, TemplateSuggester,
                                   extract_subject)


@pytest.mark.parametrize(("caption", "expected"), [
    ("A drawing of a lounge chair.", "a lounge chair"),
    ("a black and white sketch of a desk lamp", "a desk lamp"),
    ("a chair", "a chair"),
    ("a drawing of", NEUTRAL_SUBJECT),
    ("", NEUTRAL_SUBJECT),
    ("a drawing of a very tall wooden chair with a curved back", "a very tall wooden chair with"),
], ids=["framed", "black-and-white", "plain", "framing-only", "empty", "long"])
def test_extract_subject(caption: str, expected: str) -> None:
    """
    The medium words are dropped, long captions are cut to six words, and an empty result becomes the neutral subject.
    """
    subject = extract_subject(caption)
    assert subject == expected, f"Expected {expected!r} from {caption!r}, got {subject!r}"


def test_florence_suggester_fills_templates() -> None:
    """
    The named object fills every style template, in template order.
    """
    suggester = FlorenceSuggester(captioner=lambda payload: "A drawing of a lamp.")
    suggestions = suggester.suggest(b"sketch")

    # One suggestion per template, each starting with the object.
    assert len(suggestions) == len(STYLE_TEMPLATES), f"Expected one suggestion per template, got {suggestions}"
    assert all(text.startswith("a lamp, ") for text in suggestions), f"Every suggestion must name the lamp, got {suggestions}"


def test_template_suggester_needs_no_model() -> None:
    """
    The fixed suggester fills the templates with the neutral subject without reading the sketch.
    """
    suggestions = TemplateSuggester().suggest(b"")
    assert suggestions[0].startswith(NEUTRAL_SUBJECT), f"Expected the neutral subject, got {suggestions[0]!r}"


def test_florence_suggester_missing_model(tmp_path: Path) -> None:
    """
    A missing model folder fails with one line that names the download script.
    """
    with pytest.raises(SuggestionError, match="download_models.py"):
        FlorenceSuggester(tmp_path).suggest(b"sketch")
