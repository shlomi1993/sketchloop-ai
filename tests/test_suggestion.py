"""
Tests of the prompt suggesters with a fake captioner, so no model is loaded.
"""

import pytest

from pathlib import Path

from sketchloop.suggestion import (FLORENCE_MODEL, NEUTRAL_SUBJECT, STYLE_TEMPLATES, FlorenceSuggester, SuggestionError,
                                   TemplateSuggester, extract_subject)


@pytest.mark.parametrize(("caption", "expected"), [
    ("a drawing of", NEUTRAL_SUBJECT),
    ("a drawing of a very tall wooden chair with a curved back", "a very tall wooden chair with"),
], ids=["framing-only", "long"])
def test_extract_subject(caption: str, expected: str) -> None:
    """
    The medium words are dropped, long captions are cut to six words, and an empty result becomes the neutral subject.
    """
    subject = extract_subject(caption)
    assert subject == expected, f"Expected {expected!r} from {caption!r}, got {subject!r}"


def test_prompt_suggesters_fill_the_templates(tmp_path: Path) -> None:
    """
    Florence-2 fills every template with the named object and records its revision, and the templates need no model.
    """
    metadata = tmp_path / FLORENCE_MODEL.folder / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("00921df66db728a9ceb750f5eca43e5c203a2051\netag\n", encoding="utf-8")
    florence = FlorenceSuggester(tmp_path, captioner=lambda payload: "A drawing of a lamp.")
    suggestions = florence.suggest(b"sketch")

    # One suggestion per template, each starting with the object, while the templates alone use the neutral subject.
    assert len(suggestions) == len(STYLE_TEMPLATES), f"Expected one suggestion per template, got {suggestions}"
    assert all(text.startswith("a lamp, ") for text in suggestions), f"Every suggestion must name the lamp, got {suggestions}"
    templates = TemplateSuggester().suggest(b"")
    assert templates[0].startswith(NEUTRAL_SUBJECT), f"Expected the neutral subject, got {templates[0]!r}"

    # The model ID joins the repository with the downloaded revision, cut to 12 characters, and it is not a fake.
    identity = florence.identity
    expected = "florence-community/Florence-2-base@00921df66db7"
    assert identity.model_id == expected, f"Expected {expected}, got {identity.model_id}"
    assert identity.adapter == "sketchloop.florence-prompts" and not identity.is_fake, f"Unexpected identity {identity}"

    # Without a captioner, a missing model folder fails with one line that names the download script.
    with pytest.raises(SuggestionError, match="download_models.py"):
        FlorenceSuggester(tmp_path / "empty").suggest(b"sketch")
