"""
Tests of experiment name rules and the session record's optional name.
"""

import pytest

from dataclasses import replace
from datetime import datetime, timezone

from sketchloop.domain import InvalidRecordError, Unavailable
from sketchloop.experiments import EXPERIMENT_NAME_RULE, Environment, ExperimentNameError, SessionRecord, check_experiment_name


def make_session_record(session_id: str, name: str | None) -> SessionRecord:
    """
    Build a minimal session record with the given folder name and experiment name.
    """
    environment = Environment(py_version="3.12.0", platform="test", packages={}, hardware=Unavailable("test"))
    return SessionRecord(schema_version=1, session_id=session_id, created_at=datetime.now(timezone.utc), environment=environment,
                         generator_setup_seconds=Unavailable("test"), rerun_of=None, benchmark=None, name=name)


@pytest.mark.parametrize(("name", "expected"), [("  P-03 ", "P-03"), ("a" * 64, "a" * 64), ("con1", "con1")])
def test_check_experiment_name_accepts_codes(name: str, expected: str) -> None:
    """
    Codes of letters, digits, "_", and "-" pass, with surrounding spaces stripped.
    """
    checked = check_experiment_name(name)

    # The name comes back stripped and otherwise unchanged.
    assert checked == expected, f"Expected {expected!r}, got {checked!r}"


@pytest.mark.parametrize("name", ["", "../x", "a\\b", "C:x", ".hidden", "has space", "COM1", "Aux", "a" * 65, "-lead", "café",
                                  "x\x00y"])
def test_check_experiment_name_refuses_unsafe_names(name: str) -> None:
    """
    Empty names, separators, drive letters, dots, spaces, device names, non-ASCII, nul, and 65 characters fail in one line.
    """
    with pytest.raises(ExperimentNameError) as caught:
        check_experiment_name(name)

    # The message is the one-line rule, which names a valid example.
    message = str(caught.value)
    assert message == EXPERIMENT_NAME_RULE and "\n" not in message, f"Expected the one-line rule, got {message!r}"


def test_session_record_name_must_follow_rules_and_equal_folder() -> None:
    """
    A session record keeps a valid name equal to its folder, and refuses a name that differs or breaks the rules.
    """
    record = make_session_record("participant1_car", "participant1_car")

    # A matching name is kept, and a time-named session has none.
    assert record.name == "participant1_car", f"Expected the name kept, got {record.name!r}"
    assert make_session_record("20261121-100000-abcdef", None).name is None, "A time-named session must have no name"

    # A name other than the folder, or one breaking the rules, is refused.
    with pytest.raises(InvalidRecordError, match="equal the session ID"):
        replace(record, name="participant2_car")
    with pytest.raises(InvalidRecordError, match="naming rules"):
        make_session_record("../x", "../x")
