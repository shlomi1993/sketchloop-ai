import hashlib
import json
import struct
import uuid
import zlib

from collections.abc import Mapping

from sketchloop.domain import (BackendIdentity, Candidate, ControlValue, EffectiveSettings, GenerationRequest,
                               GenerationResult, ImageRef, Unavailable)
from sketchloop.generation import ControlSpec, GenerationOutput, GeneratorCapabilities, validate_request

# Controls typical of an image-to-image diffusion backend, so tests exercise realistic validation.
_DEFAULT_CONTROLS = (
    ControlSpec(name="steps", kind="int", minimum=1, maximum=50, default=4),
    ControlSpec(name="guidance_scale", kind="float", minimum=0, maximum=20, default=7.5),
    ControlSpec(name="strength", kind="float", minimum=0, maximum=1, default=0.75)
)
_DEFAULT_CAPABILITIES = GeneratorCapabilities(
    controls=_DEFAULT_CONTROLS, max_candidates=8, supports_negative_prompt=True, supports_seed=True
)


class FakeGenerator:
    """
    Deterministic test-only generator that labels its output as fake.
    """

    def __init__(self, capabilities: GeneratorCapabilities | None = None, *, size: tuple[int, int] = (256, 256)) -> None:
        self._capabilities = capabilities or _DEFAULT_CAPABILITIES
        self._size = size

    @property
    def capabilities(self) -> GeneratorCapabilities:
        """
        Return the fake backend's capabilities.
        """
        return self._capabilities

    def generate(self, request: GenerationRequest, sketch_payload: bytes) -> GenerationOutput:
        """
        Validate the request and render deterministic grayscale images derived from it.

        Args:
            request (GenerationRequest): Request to generate from.
            sketch_payload (bytes): Encoded sketch bytes, unused because the fake derives images from the request alone.

        Returns:
            GenerationOutput: Candidates and their image bytes.
        """
        # Validate like a real backend, then fill in defaults so the effective settings are complete.
        validate_request(request, self._capabilities)
        defaults = {spec.name: spec.default for spec in self._capabilities.controls if spec.default is not None}
        controls = defaults | dict(request.guidance.controls)

        # Render one image per candidate, giving each its own seed so the outputs differ.
        candidates = []
        payloads = {}
        for index in range(request.n_candidates):
            seed = (request.seed or 0) + index
            payload = self._render(request, controls, index, seed)
            candidate_id = uuid.uuid4().hex
            image = ImageRef(path=f"candidates/{candidate_id}.png", width=self._size[0], height=self._size[1], mode="L",
                             media_type="image/png", sha256=hashlib.sha256(payload).hexdigest())
            candidates.append(Candidate(id=candidate_id, index=index, image=image, seed=seed))
            payloads[image.path] = payload

        # Label the backend as fake so its output is never mistaken for a real model run.
        backend = BackendIdentity(adapter="sketchloop.fake", adapter_version="1", execution="fake",
                                  model_id=Unavailable("fake backend has no model"))
        prompt, negative_prompt = request.guidance.prompt, request.guidance.negative_prompt
        effective = EffectiveSettings(prompt=prompt, negative_prompt=negative_prompt, controls=controls)
        result = GenerationResult(candidates=tuple(candidates), backend=backend, effective=effective)
        return GenerationOutput(result=result, payloads=payloads)

    def _render(self, request: GenerationRequest, controls: Mapping[str, ControlValue], index: int, seed: int) -> bytes:
        # Hash every input that affects the output, so identical requests give identical bytes.
        material = {"sketch": request.sketch.sha256, "prompt": request.guidance.prompt, "controls": dict(controls),
                    "negative_prompt": request.guidance.negative_prompt, "index": index, "seed": seed}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).digest()

        # Stretch the digest into enough pixel bytes and encode them as a grayscale PNG.
        width, height = self._size
        pixel_count = width * height
        blocks = (hashlib.sha256(digest + block.to_bytes(4, "big")).digest() for block in range(pixel_count // 32 + 1))
        return _encode_grayscale_png(width, height, b"".join(blocks)[:pixel_count])


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _encode_grayscale_png(width: int, height: int, pixels: bytes) -> bytes:
    # Prefix each row with filter type 0 (none), then write the 8-bit grayscale header, pixel data, and end chunks.
    rows = b"".join(b"\x00" + pixels[row * width:(row + 1) * width] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunks = _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + chunks
