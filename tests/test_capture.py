"""
Tests of camera capture with a fake camera and of JPEG frame encoding.
"""

import cv2
import numpy as np
import pytest
import time

from sketchloop.capture import SPACE_KEY, CameraCapture, InvalidSketchError, VideoSource, capture_from_camera, encode_frame_jpeg


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
    Space captures the previewed frame as a PNG raw sketch and releases the camera, and a missing camera fails in one line.
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

    # A camera that cannot open gives a one-line error naming its index and still gets released.
    missing = FakeCamera(available=False)
    with pytest.raises(InvalidSketchError, match="No camera at index 2") as error:
        capture_from_camera(2, open_camera=lambda index: missing)
    assert "\n" not in str(error.value) and missing.released, f"Expected a one-line error and a release, got {error.value!r}"


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


def test_encode_frame_jpeg() -> None:
    """
    A BGRA frame is encoded as a JPEG raw frame without alpha, and a bad quality or a frame that is not 8-bit is refused.
    """
    image, payload = encode_frame_jpeg(np.full((48, 64, 4), 220, dtype=np.uint8), 90)

    # The bytes are JPEG and the reference describes them.
    assert payload.startswith(b"\xff\xd8\xff"), "Expected JPEG bytes"
    described = (image.path, image.media_type, image.width, image.height, image.mode)
    assert described == ("sketch-raw.jpg", "image/jpeg", 64, 48, "RGB"), f"Unexpected reference {described}"

    # A quality outside 1 to 100 and a 16-bit frame are refused.
    with pytest.raises(ValueError):
        encode_frame_jpeg(np.zeros((8, 8, 3), dtype=np.uint8), 0)
    with pytest.raises(InvalidSketchError):
        encode_frame_jpeg(np.zeros((8, 8, 3), dtype=np.uint16), 90)
