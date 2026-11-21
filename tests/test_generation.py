"""
Test that every unsupported requested setting is reported together.
"""

import pytest


from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import GenerationRequest, Guidance, ImageRef, UnsupportedConfigurationError
from sketchloop.generation import ControlSpec, GeneratorCapabilities, control_issues, request_issues


SKETCH = ImageRef(path="sketches/s1.png", width=8, height=8, mode="L", media_type="image/png", sha256="a" * 64)


SKETCH_BYTES = b"unused by the fake"


def test_every_unsupported_setting_is_reported_together() -> None:
    """
    A request that breaks every capability must fail once, listing all issues in order.
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


def test_control_issues_match_request_issues_under_any_prefix() -> None:
    """
    The extracted control check gives the request check's control issues, named under the caller's field prefix.
    """
    # Break every control rule once, so each kind of issue appears.
    specs = (
        ControlSpec(name="steps", kind="int", minimum=1, maximum=50),
        ControlSpec(name="sampler", kind="choice", choices=("euler", "ddim"))
    )
    controls = {"steps": 0, "sampler": "x", "eta": 1}
    capabilities = GeneratorCapabilities(controls=specs, max_candidates=1, supports_negative_prompt=True, supports_seed=True)
    request = GenerationRequest(sketch=SKETCH, guidance=Guidance(prompt="chair", controls=controls))
    from_request = request_issues(request, capabilities)
    from_controls = control_issues(controls, specs, "controls")

    # Both list the same codes and messages in name order, and only the field prefix differs.
    expected = [("eta", "unknown_control"), ("sampler", "invalid_choice"), ("steps", "out_of_range")]
    request_fields = [(issue.field, issue.code) for issue in from_request]
    control_fields = [(issue.field, issue.code) for issue in from_controls]
    assert request_fields == [(f"guidance.controls.{name}", code) for name, code in expected], f"Changed: {request_fields}"
    assert control_fields == [(f"controls.{name}", code) for name, code in expected], f"Wrong prefix: {control_fields}"
    assert [issue.message for issue in from_controls] == [issue.message for issue in from_request], "Messages must be shared"
    assert control_issues({"steps": 4}, specs, "controls") == (), "Valid controls must give no issues"
