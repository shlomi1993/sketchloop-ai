import hashlib
import json
import uuid

from collections.abc import Mapping

from sketchloop.domain import (BackendIdentity, Candidate, ControlValue, EffectiveSettings, GenerationRequest,
                               GenerationResult, ImageRef, Unavailable)
from sketchloop.generation import ControlSpec, GenerationOutput, GeneratorCapabilities, validate_request

_DEFAULT_CONTROLS = (ControlSpec(name="steps", kind="int", minimum=1, maximum=50, default=4),
                     ControlSpec(name="guidance_scale", kind="float", minimum=0, maximum=20, default=7.5),
                     ControlSpec(name="strength", kind="float", minimum=0, maximum=1, default=0.75))
_DEFAULT_CAPABILITIES = GeneratorCapabilities(controls=_DEFAULT_CONTROLS, max_candidates=8,
                                              supports_negative_prompt=True, supports_seed=True)


class FakeGenerator:
    """Deterministic test-only generator that labels its output as fake."""

    def __init__(self, capabilities: GeneratorCapabilities | None = None, *, size: tuple[int, int] = (8, 8)) -> None:
        self._capabilities = capabilities or _DEFAULT_CAPABILITIES
        self._size = size

    def capabilities(self) -> GeneratorCapabilities:
        """Return the fake backend's capabilities.

        Returns:
            GeneratorCapabilities: Supported controls and limits.
        """
        return self._capabilities

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        """Validate the request and render deterministic grayscale images derived from it.

        Args:
            request (GenerationRequest): Request to generate from.

        Returns:
            GenerationOutput: Candidates and their image bytes.
        """
        validate_request(request, self._capabilities)
        defaults = {spec.name: spec.default for spec in self._capabilities.controls if spec.default is not None}
        controls = defaults | dict(request.guidance.controls)
        candidates = []
        payloads = {}
        for index in range(request.num_candidates):
            seed = (request.seed or 0) + index
            payload = self._render(request, controls, index, seed)
            candidate_id = uuid.uuid4().hex
            image = ImageRef(path=f"candidates/{candidate_id}.pgm", width=self._size[0], height=self._size[1], mode="L",
                             media_type="image/x-portable-graymap", sha256=hashlib.sha256(payload).hexdigest())
            candidates.append(Candidate(id=candidate_id, index=index, image=image, seed=seed))
            payloads[image.path] = payload

        backend = BackendIdentity(adapter="sketchloop.fake", adapter_version="1", execution="fake",
                                  model_id=Unavailable("fake backend has no model"))
        prompt, negative_prompt = request.guidance.prompt, request.guidance.negative_prompt
        effective = EffectiveSettings(prompt=prompt, negative_prompt=negative_prompt, controls=controls)
        result = GenerationResult(candidates=tuple(candidates), backend=backend, effective=effective)
        return GenerationOutput(result=result, payloads=payloads)

    def _render(self, request: GenerationRequest, controls: Mapping[str, ControlValue], index: int, seed: int) -> bytes:
        material = {"sketch": request.sketch.sha256, "prompt": request.guidance.prompt, "controls": dict(controls),
                    "negative_prompt": request.guidance.negative_prompt, "index": index, "seed": seed}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).digest()
        width, height = self._size
        pixel_count = width * height
        blocks = (hashlib.sha256(digest + block.to_bytes(4, "big")).digest() for block in range(pixel_count // 32 + 1))
        return f"P5\n{width} {height}\n255\n".encode() + b"".join(blocks)[:pixel_count]
