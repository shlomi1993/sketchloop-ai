import hashlib
import struct

from pathlib import Path
from typing import Final

from sketchloop.domain import ColorMode, ImageRef, SketchLoopError


# PNG files start with this signature, followed by the IHDR chunk that holds size and color type.
PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
MODE_BY_COLOR_TYPE: Final[dict[int, ColorMode]] = {0: "L", 2: "RGB", 6: "RGBA"}


class InvalidSketchError(SketchLoopError, ValueError):
    """
    A sketch file is missing, unreadable, or not a supported PNG.
    """
    pass


def load_sketch_file(path: Path) -> tuple[ImageRef, bytes]:
    """
    Load a PNG sketch and describe it as an image reference relative to its run folder.

    Args:
        path (Path): PNG file to load.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the file's original bytes.
    """
    # Read the whole file, since it is copied into the run folder unchanged.
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise InvalidSketchError(f"Cannot read sketch {path.name}: {error.strerror}. Check the path.") from error

    # Split out the PNG layout: signature, first chunk type, then the IHDR width, height, bit depth, and color type.
    signature = payload[:8]
    first_chunk_type = payload[12:16]
    header_fields = payload[16:26]

    # Accept only PNG files with a complete IHDR chunk.
    if signature != PNG_SIGNATURE or first_chunk_type != b"IHDR" or len(payload) < 33:
        raise InvalidSketchError(f"Sketch {path.name} is not a PNG file. Save it as PNG and retry.")

    # Map the PNG color type to a mode, rejecting palette, gray-alpha, and non-8-bit images.
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header_fields)
    if color_type not in MODE_BY_COLOR_TYPE.keys() or bit_depth != 8:
        raise InvalidSketchError(f"Sketch {path.name} uses an unsupported PNG format. Save it as 8-bit grayscale, RGB, or RGBA.")

    image = ImageRef(
        path="sketch.png",
        width=width,
        height=height,
        mode=MODE_BY_COLOR_TYPE[color_type],
        media_type="image/png",
        sha256=hashlib.sha256(payload).hexdigest()
    )
    return image, payload
