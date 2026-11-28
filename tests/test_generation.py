"""
Test that every unsupported requested setting is reported together.
"""

import pytest


from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import GenerationRequest, Guidance, ImageRef, UnsupportedConfigurationError
from sketchloop.generation import ControlSpec, GeneratorCapabilities, control_issues


SKETCH = ImageRef(path="sketches/s1.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


SKETCH_BYTES = b"unused by the fake"


def test_every_unsupported_setting_is_reported_together() -> None:
    """
    A request that breaks every capability fails once, listing all issues in order, as the shared control check does.
    """
    # Build a restrictive backend and a request that breaks every rule it has.
    controls = (
        ControlSpec(name="steps", kind="int", minimum=1, maximum=50),
        ControlSpec(name="scale", kind="float", minimum=0, maximum=20),
        ControlSpec(name="sampler", kind="choice", choices=("euler", "ddim"))
    )
    capabilities = GeneratorCapabilities(controls=controls, max_candidates=8, supports_negative_prompt=False, supports_seed=False)
    invalid = {"steps": 51, "scale": True, "sampler": "x", "eta": 1}
    guidance = Guidance(prompt="chair", negative_prompt="", controls=invalid)
    request = GenerationRequest(sketch=SKETCH, guidance=guidance, n_candidates=9, seed=0)
    with pytest.raises(UnsupportedConfigurationError) as raised:
        FakeGenerator(capabilities).generate(request, SKETCH_BYTES)

    # Request-level issues come first, then controls in name order.
    expected = [
        ("n_candidates", "too_many_candidates"),
        ("seed", "unsupported_feature"),
        ("guidance.negative_prompt", "unsupported_feature"),
        ("guidance.controls.eta", "unknown_control"),
        ("guidance.controls.sampler", "invalid_choice"),
        ("guidance.controls.scale", "wrong_type"),
        ("guidance.controls.steps", "out_of_range")
    ]
    actual = [(issue.field, issue.code) for issue in raised.value.issues]
    assert actual == expected, f"Every issue must be reported in order, got {actual}"

    # The shared control check gives the same control issues and messages under the caller's prefix, and none when valid.
    from_controls = control_issues(invalid, controls, "controls")
    control_fields = [(issue.field, issue.code) for issue in from_controls]
    assert control_fields == [(field.replace("guidance.", ""), code) for field, code in expected[3:]], f"Got {control_fields}"
    assert [issue.message for issue in from_controls] == [issue.message for issue in raised.value.issues[3:]], "Bad messages"
    assert control_issues({"steps": 4}, controls, "controls") == (), "Valid controls must give no issues"
