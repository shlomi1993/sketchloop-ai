"""
Tests of file and camera capture with a fake camera.
"""

import cv2
import numpy as np
import pytest
import time

from pathlib import Path

from sketchloop.capture import (SPACE_KEY, CameraCapture, CapturedSketch, FileCapture, InvalidSketchError, VideoSource,
                                capture_from_camera, load_sketch_file)


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


class FakeCamera(VideoSource):
    """
    Test camera that serves one synthetic frame, or none when unavailable.
    """

    def __init__(self, available: bool) -> None:
        self.available = available
        self.released = False

    def is_opened(self) -> bool:
        """
        Report whether the fake camera is available.
        """
        return self.available

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return a synthetic light frame with a dark stroke.
        """
        frame = np.full((48, 64, 3), 220, dtype=np.uint8)
        cv2.line(frame, (5, 5), (50, 40), (20, 20, 20), 2)
        return True, frame

    def release(self) -> None:
        """
        Record that the camera was released.
        """
        self.released = True


@pytest.fixture
def no_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace OpenCV window calls so tests show nothing and Space is pressed on the first frame.
    """
    monkeypatch.setattr(cv2, "imshow", lambda title, frame: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: SPACE_KEY)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda title, prop: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


@pytest.mark.usefixtures("no_windows")
def test_camera_capture() -> None:
    """
    Pressing Space captures the previewed frame as a PNG raw sketch and releases the camera, also through CameraCapture.
    """
    camera = FakeCamera(available=True)
    source = CameraCapture(0, open_camera=lambda index: camera)

    captured = source.capture()
    image, payload = captured.image, captured.payload
    direct = capture_from_camera(0, open_camera=lambda index: FakeCamera(available=True))

    # The capture is a PNG raw sketch with the frame's size, and the camera is released.
    image_ref_is_raw_png = image.path == "sketch-raw.png" and image.media_type == "image/png" and payload.startswith(b"\x89PNG")
    assert image_ref_is_raw_png, f"Expected a PNG raw sketch, got {image}"
    assert (image.width, image.height, image.mode) == (64, 48, "RGB"), f"Unexpected size or mode {image}"
    assert camera.released, "The camera must be released after capture"

    # The capture source records camera input and gives the same sketch as capturing directly.
    assert source.kind == "camera", f"CameraCapture must record source camera, got {source.kind}"
    assert (image, payload) == (direct.image, direct.payload), "CameraCapture must return the same sketch as capture_from_camera"


@pytest.mark.usefixtures("no_windows")
def test_camera_unavailable() -> None:
    """
    A camera that cannot open gives a one-line error naming the index and still gets released.
    """
    camera = FakeCamera(available=False)

    # Opening the missing camera fails with an error naming its index.
    with pytest.raises(InvalidSketchError, match="No camera at index 2") as error:
        capture_from_camera(2, open_camera=lambda index: camera)

    # The error fits on one line, and the camera is still released.
    assert "\n" not in str(error.value), f"Expected a one-line error, got {error.value!r}"
    assert camera.released, "The camera must be released after a failure"


def test_file_capture_is_timed() -> None:
    """
    A file capture returns the file's sketch with a non-negative capture time.
    """
    captured = FileCapture(EXAMPLE_SKETCH).capture()

    # The capture holds the same sketch as loading the file, with its own time.
    assert isinstance(captured, CapturedSketch), f"Expected a CapturedSketch, got {type(captured)}"
    assert (captured.image, captured.payload) == load_sketch_file(EXAMPLE_SKETCH), "The capture must hold the file's sketch"
    assert captured.capture_seconds >= 0, f"Capture time must not be negative, got {captured.capture_seconds}"


@pytest.mark.usefixtures("no_windows", "counting_clock")
def test_camera_capture_excludes_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A camera capture starts its clock when Space is pressed, so 100 s spent looking at the preview are not counted.
    """
    def preview_for_100_seconds(camera: VideoSource, unavailable: str) -> np.ndarray:
        """
        Stand in for a person looking at the preview for 100 s before pressing Space.
        """
        time.perf_counter.advance(100.0)
        return camera.read()[1]

    monkeypatch.setattr("sketchloop.capture.preview_until_capture", preview_for_100_seconds)
    captured = capture_from_camera(0, open_camera=lambda index: FakeCamera(available=True))

    # Only the encode after Space is timed, which is one 7 s clock step, never the preview.
    assert captured.capture_seconds == 7.0, f"The preview must not count as capture time, got {captured.capture_seconds}"
