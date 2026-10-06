import cv2
import hashlib
import numpy as np

from pathlib import Path
from typing import Final

from sketchloop.domain import ColorMode, ImageRef, SketchLoopError


# Supported formats by file signature, with the raw file extension and media type to store them under.
FORMAT_BY_SIGNATURE: Final = {
    b"\x89PNG\r\n\x1a\n": (".png", "image/png"),
    b"\xff\xd8\xff": (".jpg", "image/jpeg")
}
MODE_BY_CHANNELS: Final[dict[int, ColorMode]] = {1: "L", 3: "RGB", 4: "RGBA"}


class InvalidSketchError(SketchLoopError, ValueError):
    """
    A sketch file is missing, unreadable, or not a supported PNG or JPEG.
    """
    pass


def load_sketch_file(path: Path) -> tuple[ImageRef, bytes]:
    """
    Load a PNG or JPEG sketch and describe it as an image reference relative to its run folder.

    Args:
        path (Path): PNG or JPEG file to load.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the file's original bytes.
    """
    # Read the whole file, since it is copied into the run folder unchanged.
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise InvalidSketchError(f"Cannot read sketch {path.name}: {error.strerror}. Check the path.") from error

    # Identify the format from the file signature rather than trusting the extension.
    matches = [file_format for signature, file_format in FORMAT_BY_SIGNATURE.items() if payload.startswith(signature)]
    if not matches:
        raise InvalidSketchError(f"Sketch {path.name} is not a PNG or JPEG file. Save it as PNG or JPEG and retry.")

    # Decode from memory, so paths with non-ASCII characters work, and accept only 8-bit images.
    extension, media_type = matches[0]
    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    channels = 0 if pixels is None else 1 if pixels.ndim == 2 else pixels.shape[2]
    if channels not in MODE_BY_CHANNELS or pixels.dtype != np.uint8:
        raise InvalidSketchError(f"Sketch {path.name} cannot be decoded as 8-bit gray, RGB, or RGBA. Re-save it as PNG or JPEG.")

    image = ImageRef(
        path=f"sketch-raw{extension}",
        width=pixels.shape[1],
        height=pixels.shape[0],
        mode=MODE_BY_CHANNELS[channels],
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest()
    )
    return image, payload
