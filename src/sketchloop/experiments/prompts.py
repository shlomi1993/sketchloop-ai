"""
Prompt suggestion records, and how the prompt sent with a stroke suggestion, a round, or a finish came about.
"""

import math

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from sketchloop.domain import BackendIdentity, Unavailable, require
from sketchloop.stroke_suggestion import is_count


ORIGIN_NOT_RECORDED: Final = "not recorded before prompt origins were"
OUTSIDE_DRAWING_REASON: Final = "the page named a prompt suggestion outside this drawing"
OUTSIDE_SESSION_REASON: Final = "the page named a prompt suggestion outside this session"
SHOWN_LATER_REASON: Final = "the page named a prompt suggestion shown after this prompt was used"


class PromptOriginKind(StrEnum):
    TYPED = "typed"  # No suggestion was in the box, or the prompt was blank
    SUGGESTED = "suggested"  # The prompt equals the suggestion written into the box
    EDITED = "edited"  # A suggestion was written into the box and the prompt differs from it


class PromptSuggestionStatus(StrEnum):
    SHOWN = "shown"
    FAILED = "failed"


def is_aware_time(moment: object) -> bool:
    return isinstance(moment, datetime) and moment.utcoffset() is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptOrigin:
    kind: PromptOriginKind  # typed, suggested, or edited
    prompt_suggestion: str | None  # ID of the suggestion the page wrote into its prompt box, None when typed

    def __post_init__(self) -> None:
        # Hold a known kind as the enum, and name a suggestion exactly when the prompt came from one.
        kinds = ", ".join(PromptOriginKind)
        require(self.kind in list(PromptOriginKind), f"Prompt origin kind must be one of {kinds}, got {self.kind!r}.")
        object.__setattr__(self, "kind", PromptOriginKind(self.kind))
        names_one = isinstance(self.prompt_suggestion, str) and bool(self.prompt_suggestion.strip())
        require(self.prompt_suggestion is None or names_one, "Prompt origin must name its suggestion by a non-empty ID.")
        require(names_one == (self.kind != PromptOriginKind.TYPED), f"Prompt origin {self.kind} names a suggestion exactly when not typed.")  # noqa: E501


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggestionFailure:
    error: str  # Error class name
    message: str  # First line of the error message, with paths redacted


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggestionRecord:
    number: int  # Position in the drawing's or round's list, from 1
    id: str  # Unique ID the page sends back with the prompt this suggestion was written into
    status: PromptSuggestionStatus  # shown, or failed with a failure record
    requested_at: datetime  # Timezone-aware time of the press
    shown_at: datetime | None  # Timezone-aware time the text reached the page, None when failed
    suggester: BackendIdentity  # The prompt suggester's identity
    text: str | None  # Suggested prompt written into the box, None when failed
    list_position: int | None  # 0-based position in the suggester's list, such as the style template, None when failed
    ran_suggester: bool  # True when this press ran the suggester, false when it took the next entry for an unchanged sketch
    suggester_seconds: float | None  # Seconds the suggester ran, None when it did not run
    after_capture: int | None  # Latest capture of the drawing when shown, None outside a drawing or before its first capture
    failure: PromptSuggestionFailure | None  # Why it failed, None when shown

    def __post_init__(self) -> None:
        # Number, ID, status, press time, and suggester identify the press.
        number = self.number
        require(is_count(number, 1), f"Prompt suggestion number must be an int >= 1, got {number!r}.")
        require(isinstance(self.id, str) and bool(self.id.strip()), f"Prompt suggestion {number} needs a non-empty ID.")
        status = self.status
        require(status in list(PromptSuggestionStatus), f"Prompt suggestion status must be shown or failed, got {status!r}.")
        object.__setattr__(self, "status", PromptSuggestionStatus(status))
        require(is_aware_time(self.requested_at), f"Prompt suggestion {number} requested_at must be timezone-aware.")
        require(isinstance(self.suggester, BackendIdentity), f"Prompt suggestion {number} needs its suggester's identity.")
        capture = self.after_capture
        require(capture is None or is_count(capture, 1), f"Prompt suggestion {number} after_capture must be an int >= 1 or null.")

        # Seconds are recorded exactly when the suggester ran.
        seconds = self.suggester_seconds
        require(isinstance(self.ran_suggester, bool), f"Prompt suggestion {number} ran_suggester must be true or false.")
        timed = isinstance(seconds, int | float) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0
        require(timed if self.ran_suggester else seconds is None, f"Prompt suggestion {number} has finite suggester_seconds exactly when it ran the suggester.")  # noqa: E501
        self._check_outcome()

    def _check_outcome(self) -> None:
        # A shown suggestion has its time shown, text, and list position, and a failed one has only its failure.
        number = self.number
        shown = self.status == PromptSuggestionStatus.SHOWN
        require(shown == (self.failure is None), f"Prompt suggestion {number} must be shown exactly when it has no failure.")
        if not shown:
            require((self.shown_at, self.text, self.list_position) == (None, None, None), f"Failed prompt suggestion {number} cannot have a time shown, text, or list position.")  # noqa: E501
            return

        require(isinstance(self.text, str) and bool(self.text.strip()), f"Shown prompt suggestion {number} needs its text.")
        require(is_count(self.list_position, 0), f"Prompt suggestion {number} list_position must be an int >= 0.")
        valid_time = is_aware_time(self.shown_at) and self.shown_at >= self.requested_at
        require(valid_time, f"Prompt suggestion {number} shown_at must be timezone-aware and not before requested_at.")


def classify_prompt_origin(prompt: str, suggestion: PromptSuggestionRecord | None) -> PromptOrigin:
    # A blank prompt, or one with no suggestion in the box, was typed, and otherwise the texts are compared without spaces.
    if suggestion is None or not prompt.strip():
        return PromptOrigin(kind=PromptOriginKind.TYPED, prompt_suggestion=None)

    require(suggestion.text is not None, f"Prompt suggestion {suggestion.number} failed, so no prompt can come from it.")
    same = prompt.strip() == suggestion.text.strip()
    return PromptOrigin(kind=PromptOriginKind.SUGGESTED if same else PromptOriginKind.EDITED, prompt_suggestion=suggestion.id)


def find_shown_suggestion(suggestions: Iterable[PromptSuggestionRecord], suggestion_id: str) -> PromptSuggestionRecord | None:
    # Only a shown suggestion could have been written into the prompt box.
    return next((item for item in suggestions if item.id == suggestion_id and item.status == PromptSuggestionStatus.SHOWN), None)


def resolve_prompt_origin(prompt: str, suggestion_id: str | None, suggestions: Iterable[PromptSuggestionRecord],
                          used_at: datetime, missing_reason: str) -> PromptOrigin | Unavailable:
    # No ID means nothing was written into the box, so the prompt was typed.
    if suggestion_id is None:
        return classify_prompt_origin(prompt, None)

    # An ID from a stale tab is not found, and is recorded as unavailable rather than blocking the use.
    suggestion = find_shown_suggestion(suggestions, suggestion_id)
    if suggestion is None:
        return Unavailable(missing_reason)

    # A suggestion shown after the use cannot have been in the box, which only a clock change could cause.
    if suggestion.shown_at > used_at:
        return Unavailable(SHOWN_LATER_REASON)

    return classify_prompt_origin(prompt, suggestion)


def check_prompt_suggestions(suggestions: tuple[PromptSuggestionRecord, ...], owner: str) -> None:
    # Suggestions are numbered from 1 without gaps, with unique IDs, in the order they were pressed.
    numbers = [suggestion.number for suggestion in suggestions]
    require(numbers == list(range(1, len(numbers) + 1)), f"{owner} prompt suggestion numbers must run from 1 without gaps, got {numbers}.")  # noqa: E501
    ids = [suggestion.id for suggestion in suggestions]
    require(len(set(ids)) == len(ids), f"{owner} prompt suggestion IDs must be unique.")
    times = [suggestion.requested_at for suggestion in suggestions]
    require(times == sorted(times), f"{owner} prompt suggestions must be in the order they were requested.")


def check_prompt_origin(origin: PromptOrigin | Unavailable, prompt: str | None, used_at: datetime,
                        suggestions: Iterable[PromptSuggestionRecord], label: str) -> None:
    # An origin naming a suggestion names a shown one from before the use, whose comparison with the prompt gives its kind.
    if isinstance(origin, Unavailable) or origin.prompt_suggestion is None:
        return

    suggestion = find_shown_suggestion(suggestions, origin.prompt_suggestion)
    require(suggestion is not None, f"{label} names prompt suggestion {origin.prompt_suggestion}, which is not a shown one of its record.")  # noqa: E501
    number = suggestion.number
    require(suggestion.shown_at <= used_at, f"{label} names prompt suggestion {number}, shown after its prompt was used.")
    if prompt is not None:
        kind = classify_prompt_origin(prompt, suggestion).kind
        require(kind == origin.kind, f"{label} is recorded as {origin.kind}, but its prompt and suggestion give {kind}.")
