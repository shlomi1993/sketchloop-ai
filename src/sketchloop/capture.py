import cv2
import hashlib
import numpy as np
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol

from sketchloop.domain import ColorMode, ImageRef, SketchLoopError


# Supported formats by file signature, with the raw file extension and media type to store them under.
FORMAT_BY_SIGNATURE: Final = {
    b"\x89PNG\r\n\x1a\n": (".png", "image/png"),
    b"\xff\xd8\xff": (".jpg", "image/jpeg")
}
MODE_BY_CHANNELS: Final[dict[int, ColorMode]] = {1: "L", 3: "RGB", 4: "RGBA"}

# Camera preview window title and the keys that capture or cancel, as returned by cv2.waitKey.
PREVIEW_WINDOW_TITLE: Final = "sketchloop - Space to capture, Esc to cancel"
SPACE_KEY: Final = 32
ESC_KEY: Final = 27
PREVIEW_FRAME_MILLISECONDS: Final = 30

# Native camera backend per platform: DirectShow opens much faster on Windows, AVFoundation is the macOS camera API.
CAMERA_BACKEND_BY_PLATFORM: Final = {"win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION}

# Some cameras (notably on macOS) return empty frames briefly after opening, so allow a short warm-up.
WARM_UP_ATTEMPTS: Final = 30


class InvalidSketchError(SketchLoopError, ValueError):
    """
    A sketch file or camera capture is missing, unreadable, cancelled, or not a supported image.
    """


class VideoSource(Protocol):
    """
    The part of cv2.VideoCapture that camera capture uses, so tests can inject a fake camera.
    """

    def isOpened(self) -> bool:
        """
        Report whether the camera opened.

        Returns:
            bool: True when frames can be read.
        """
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        Returns:
            tuple[bool, np.ndarray | None]: Whether a frame was read, and the BGR frame.
        """
        ...

    def release(self) -> None:
        """
        Release the camera.
        """
        ...


def count_channels(pixels: np.ndarray) -> int:
    """
    Count an image's color channels, where a two-dimensional array has one.
    """
    return 1 if pixels.ndim == 2 else pixels.shape[2]


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
    if pixels is None or count_channels(pixels) not in MODE_BY_CHANNELS or pixels.dtype != np.uint8:
        raise InvalidSketchError(f"Sketch {path.name} cannot be decoded as 8-bit gray, RGB, or RGBA. Re-save it as PNG or JPEG.")

    return raw_image_ref(pixels, payload, extension, media_type), payload


def raw_image_ref(pixels: np.ndarray, payload: bytes, extension: str, media_type: str) -> ImageRef:
    """
    Describe decoded raw sketch pixels and their encoded bytes as the run's raw sketch reference.

    Args:
        pixels (np.ndarray): Decoded 8-bit image with 1, 3, or 4 channels.
        payload (bytes): Encoded image bytes stored for the run.
        extension (str): File extension, including the dot.
        media_type (str): Media type of the encoded bytes.

    Returns:
        ImageRef: Reference to `sketch-raw<extension>` with size, mode, and checksum.
    """
    return ImageRef(
        path=f"sketch-raw{extension}",
        width=pixels.shape[1],
        height=pixels.shape[0],
        mode=MODE_BY_CHANNELS[count_channels(pixels)],
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest()
    )


def open_webcam(camera_index: int) -> VideoSource:
    """
    Open a camera with the native OpenCV backend: DirectShow on Windows, AVFoundation on macOS, the default elsewhere.
    """
    return cv2.VideoCapture(camera_index, CAMERA_BACKEND_BY_PLATFORM.get(sys.platform, cv2.CAP_ANY))


def capture_from_camera(camera_index: int, open_camera: Callable[[int], VideoSource] = open_webcam) -> tuple[ImageRef, bytes]:
    """
    Show a live camera preview, capture a frame on Space, and describe it as a PNG raw sketch.

    Args:
        camera_index (int): Camera index, 0 for the default camera.
        open_camera (Callable[[int], VideoSource], optional): Opens a camera by index. Defaults to OpenCV.

    Returns:
        tuple[ImageRef, bytes]: The image reference and the captured frame's PNG bytes.
    """
    camera = open_camera(camera_index)

    # Always release the camera and close the preview, even when capture fails or is cancelled.
    try:
        frame = preview_until_capture(camera, camera_unavailable_message(camera_index))
    finally:
        camera.release()
        cv2.destroyAllWindows()

        # macOS only closes windows after the event loop runs once more.
        cv2.waitKey(1)

    # Encode the captured frame losslessly, since it is the raw sketch kept for the run.
    encoded, buffer = cv2.imencode(".png", frame)
    if not encoded or count_channels(frame) not in MODE_BY_CHANNELS or frame.dtype != np.uint8:
        raise InvalidSketchError(f"Camera frame from index {camera_index} cannot be saved as PNG. Try another camera.")

    payload = buffer.tobytes()
    return raw_image_ref(frame, payload, ".png", "image/png"), payload


def camera_unavailable_message(camera_index: int) -> str:
    """
    Build the one-line error for a missing camera, with a permission hint on macOS.

    Args:
        camera_index (int): Camera index that failed.

    Returns:
        str: The error message.
    """
    message = f"No camera at index {camera_index}. Check the connection or try --camera-index 1."
    if sys.platform == "darwin":
        message += " On macOS, allow camera access for your terminal in System Settings > Privacy & Security > Camera."

    return message


def preview_until_capture(camera: VideoSource, unavailable: str) -> np.ndarray:
    """
    Show camera frames until Space captures one, tolerating a short warm-up and cancelling on Esc or a closed window.

    Args:
        camera (VideoSource): Opened camera to read from.
        unavailable (str): Error message used when the camera gives no frames.

    Returns:
        np.ndarray: The captured frame.
    """
    empty_reads = 0
    has_shown_frame = False
    while True:
        # Skip empty frames during warm-up, but fail once the camera was lost or never delivered a frame.
        grabbed, frame = camera.read() if camera.isOpened() else (False, None)
        if not grabbed or frame is None:
            empty_reads += 1
            if has_shown_frame or not camera.isOpened() or empty_reads >= WARM_UP_ATTEMPTS:
                raise InvalidSketchError(unavailable)

            cv2.waitKey(PREVIEW_FRAME_MILLISECONDS)
            continue

        # Show the frame and capture it when Space is pressed.
        cv2.imshow(PREVIEW_WINDOW_TITLE, frame)
        has_shown_frame = True
        key = cv2.waitKey(PREVIEW_FRAME_MILLISECONDS) & 0xFF
        if key == SPACE_KEY:
            return frame

        # Closing the window also cancels, except on macOS where OpenCV does not report window visibility reliably.
        window_closed = sys.platform != "darwin" and cv2.getWindowProperty(PREVIEW_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
        if key == ESC_KEY or window_closed:
            raise InvalidSketchError("Camera capture cancelled.")
