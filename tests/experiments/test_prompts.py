"""
Tests of prompt suggestion records, prompt origins, and their JSON form.
"""

import json
import pytest

from collections.abc import Callable
from dataclasses import replace

from sketchloop.domain import InvalidRecordError, Unavailable
from sketchloop.experiments import (ORIGIN_NOT_RECORDED, PromptOrigin, PromptOriginKind, PromptSuggestionFailure,
                                    PromptSuggestionRecord, PromptSuggestionStatus, classify_prompt_origin,
                                    prompt_origin_from_dict, prompt_origin_to_dict, prompt_suggestion_from_dict,
                                    prompt_suggestion_to_dict, resolve_prompt_origin)


SUGGESTED_TEXT = "the sketched object, product design render, studio lighting, white background"
PromptMaker = Callable[..., PromptSuggestionRecord]  # Builds a shown prompt suggestion from its text and ID


@pytest.mark.parametrize(("prompt", "has_suggestion", "expected"), [
    ("a red car", False, PromptOriginKind.TYPED),
    ("", True, PromptOriginKind.TYPED),
    (f"  {SUGGESTED_TEXT} ", True, PromptOriginKind.SUGGESTED),
    (SUGGESTED_TEXT.replace("studio", "soft"), True, PromptOriginKind.EDITED),
], ids=["no-suggestion", "blank", "same-text", "one-word"])
def test_classify_prompt_origin(make_prompt_suggestion: PromptMaker, prompt: str, has_suggestion: bool,
                                expected: PromptOriginKind) -> None:
    """
    A prompt matching the suggestion is suggested, a changed one edited, and a blank one or no suggestion typed.
    """
    suggestion = make_prompt_suggestion(SUGGESTED_TEXT) if has_suggestion else None
    origin = classify_prompt_origin(prompt, suggestion)

    # The kind is as expected, and only a suggested or edited prompt names the suggestion.
    assert origin.kind == expected, f"Expected {expected} for {prompt!r}, got {origin.kind}"
    expected_id = None if expected == PromptOriginKind.TYPED else "prompt-1"
    assert origin.prompt_suggestion == expected_id, f"Expected suggestion {expected_id}, got {origin.prompt_suggestion}"


@pytest.mark.parametrize(("kind", "suggestion_id"), [
    (PromptOriginKind.TYPED, "prompt-1"),
    (PromptOriginKind.EDITED, None),
    ("pasted", "prompt-1"),
])
def test_prompt_origin_refuses_inconsistent_values(kind: str, suggestion_id: str | None) -> None:
    """
    A typed origin naming a suggestion, an edited one without it, or an unknown kind is refused.
    """
    with pytest.raises(InvalidRecordError, match="Prompt origin"):
        PromptOrigin(kind=kind, prompt_suggestion=suggestion_id)


def fail_suggestion(record: PromptSuggestionRecord) -> PromptSuggestionRecord:
    failure = PromptSuggestionFailure(error="SuggestionError", message="Model folder is missing.")
    return replace(record, status=PromptSuggestionStatus.FAILED, shown_at=None, text=None, list_position=None, failure=failure)


@pytest.mark.parametrize(("change", "message"), [
    ({"text": None}, "needs its text"),
    ({"list_position": -1}, "list_position"),
    ({"ran_suggester": False}, "suggester_seconds exactly when"),
    ({"suggester_seconds": float("nan")}, "suggester_seconds"),
    ({"after_capture": 0}, "after_capture"),
    ({"failure": PromptSuggestionFailure(error="E", message="m")}, "exactly when it has no failure"),
])
def test_shown_prompt_suggestion_refuses_bad_values(make_prompt_suggestion: PromptMaker, change: dict[str, object],
                                                    message: str) -> None:
    """
    A shown prompt suggestion without text, with a bad position, timing, or capture, or with a failure is refused.
    """
    with pytest.raises(InvalidRecordError, match=message):
        replace(make_prompt_suggestion(SUGGESTED_TEXT), **change)


def test_failed_prompt_suggestion_refuses_text(make_prompt_suggestion: PromptMaker) -> None:
    failed = fail_suggestion(make_prompt_suggestion(SUGGESTED_TEXT))

    # The failed record holds no text, and adding text or a shown time breaks it.
    assert failed.text is None and failed.failure is not None, f"Expected a failed record, got {failed}"
    with pytest.raises(InvalidRecordError, match="Failed prompt suggestion"):
        replace(failed, text=SUGGESTED_TEXT)
    with pytest.raises(InvalidRecordError, match="Failed prompt suggestion"):
        replace(failed, shown_at=failed.requested_at)


def test_prompt_suggestion_round_trips(make_prompt_suggestion: PromptMaker) -> None:
    """
    Shown and failed records and every origin survive JSON, and a missing origin loads as not recorded.
    """
    shown = make_prompt_suggestion(SUGGESTED_TEXT)
    failed = fail_suggestion(replace(shown, ran_suggester=False, suggester_seconds=None))
    origins = [classify_prompt_origin("a car", None), classify_prompt_origin("a car", shown), Unavailable("a reason")]

    # Each record and origin survives a trip through JSON text.
    for record in (shown, failed):
        text = json.dumps(prompt_suggestion_to_dict(record), allow_nan=False)
        assert prompt_suggestion_from_dict(json.loads(text)) == record, f"The record must load back unchanged: {text}"
    for origin in origins:
        text = json.dumps(prompt_origin_to_dict(origin))
        assert prompt_origin_from_dict(json.loads(text)) == origin, f"The origin must load back unchanged: {text}"
    assert prompt_origin_from_dict(None) == Unavailable(ORIGIN_NOT_RECORDED), "A missing origin must load as not recorded"

    # The template suggester is recorded as a real in-process suggester without a model.
    data = prompt_suggestion_to_dict(shown)
    expected = {"adapter": "sketchloop.prompt-templates", "adapter_version": "1", "execution": "in_process",
                "model_id": {"unavailable": "fixed style templates, no model"}}
    assert data["suggester"] == expected, f"Unexpected suggester {data['suggester']}"


def test_resolve_prompt_origin(make_prompt_suggestion: PromptMaker) -> None:
    shown = make_prompt_suggestion(SUGGESTED_TEXT)
    failed = fail_suggestion(make_prompt_suggestion(SUGGESTED_TEXT, "prompt-2", number=2))
    at = shown.shown_at

    # The shown suggestion classifies the prompt, and no ID means typed.
    resolved = resolve_prompt_origin(SUGGESTED_TEXT, "prompt-1", (shown, failed), at, "outside")
    assert resolved.kind == PromptOriginKind.SUGGESTED, f"Expected suggested, got {resolved}"
    typed = resolve_prompt_origin("a car", None, (shown,), at, "outside")
    assert typed == PromptOrigin(kind=PromptOriginKind.TYPED, prompt_suggestion=None), f"Expected typed, got {typed}"

    # An unknown ID or a failed suggestion is outside, and one shown after the use is refused as later.
    unknown = [resolve_prompt_origin("a car", unknown_id, (shown, failed), at, "outside") for unknown_id in ("x", "prompt-2")]
    assert unknown == [Unavailable("outside")] * 2, f"Expected the outside reason, got {unknown}"
    earlier = resolve_prompt_origin("a car", "prompt-1", (shown,), shown.requested_at, "outside")
    assert isinstance(earlier, Unavailable) and "after" in earlier.reason, f"Expected a later-suggestion reason, got {earlier}"
