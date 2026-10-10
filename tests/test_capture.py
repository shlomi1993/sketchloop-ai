import cv2
import numpy as np
import pytest

from sketchloop.capture import SPACE_KEY, InvalidSketchError, capture_from_camera


class FakeCamera:
    """
    Test double for cv2.VideoCapture that serves one synthetic frame, or none when unavailable.
    """

    def __init__(self, available: bool) -> None:
        self.available = available
        self.released = False

    def isOpened(self) -> bool:
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
    Pressing Space captures the previewed frame as a PNG raw sketch and releases the camera.
    """
    camera = FakeCamera(available=True)

    image, payload = capture_from_camera(0, open_camera=lambda index: camera)

    # The capture is a PNG raw sketch with the frame's size, and the camera is released.
    image_ref_is_raw_png = image.path == "sketch-raw.png" and image.media_type == "image/png" and payload.startswith(b"\x89PNG")
    assert image_ref_is_raw_png, f"Expected a PNG raw sketch, got {image}"
    assert (image.width, image.height, image.mode) == (64, 48, "RGB"), f"Unexpected size or mode {image}"
    assert camera.released, "The camera must be released after capture"


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
