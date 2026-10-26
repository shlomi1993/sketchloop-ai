import numpy as np
import pytest
import re
import socket

from pathlib import Path

from sketchloop.app import NO_SKETCH_MESSAGE, AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import Rotation
from sketchloop.experiments import FolderStore


@pytest.fixture
def controller(tmp_path: Path) -> AppController:
    """
    Provide a controller with the deterministic fake backend and a runs folder under tmp_path.
    """
    return AppController(FakeGenerator(), FolderStore(tmp_path / "runs"))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make every non-loopback socket connection fail and disable Gradio analytics, so a build that needs the network fails.
    """
    connect = socket.socket.connect

    def refuse_remote_connection(sock: socket.socket, address: object) -> None:
        """
        Refuse a connection unless it targets loopback, which the Windows asyncio event loop uses internally.
        """
        # Allow loopback, since asyncio builds its self-pipe with a local socket pair.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
            return connect(sock, address)

        raise OSError(f"Network access is blocked in this test, refused {address!r}.")

    monkeypatch.setattr(socket.socket, "connect", refuse_remote_connection)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")


@pytest.mark.usefixtures("no_network")
def test_ui_builds_offline(controller: AppController, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The page builds offline without launching, analytics are off, and Generate waits for a sketch and prompt.
    """
    import gradio

    from sketchloop import ui

    def refuse_launch(*args: object, **kwargs: object) -> None:
        """
        Fail the test if the app is launched.
        """
        raise AssertionError("Building the page must not launch it")

    monkeypatch.setattr(gradio.Blocks, "launch", refuse_launch)
    app = ui.SketchPage(controller, default_candidates=2, default_seed=None).build()

    # The app is a Gradio Blocks object with analytics disabled.
    assert isinstance(app, gradio.Blocks), f"Expected gr.Blocks, got {type(app)}"
    assert app.analytics_enabled is False, "The UI must be built with analytics disabled"

    # Generate is enabled only when the chosen source has a sketch and the prompt is not blank.
    readiness = [ui.can_generate("webcam", True, None, "chair"), ui.can_generate("upload", True, None, "chair"),
                 ui.can_generate("webcam", True, None, "  "), ui.can_generate("upload", False, "b.png", "chair")]
    assert readiness == [True, False, False, True], f"Unexpected Generate readiness {readiness}"

    # The raw sketch always gets rotation 0, which the controller accepts, while each source keeps its default otherwise.
    rotations = [ui.pick_rotation("webcam", False), ui.pick_rotation("upload", False), ui.pick_rotation("webcam", True)]
    assert rotations == [180, 0, 0], f"Unexpected rotations {rotations}"

    # The stylesheet turns the preview for every rotation but none, and the script starts the webcam on load by itself.
    stylesheet = ui.STYLESHEET_PATH.read_text(encoding="utf-8")
    missing = [rotation for rotation in Rotation if rotation and f"#webcam.rotate-{rotation} " not in stylesheet]
    assert not missing, f"ui.css has no preview rule for rotations {missing}"
    assert 'addEventListener("load"' in ui.SCRIPT_PATH.read_text(encoding="utf-8"), "ui.js must start the webcam on load"


@pytest.mark.usefixtures("no_network")
def test_page_handlers_keep_frame_and_refuse_missing_sketch(controller: AppController) -> None:
    """
    SketchPage handlers work without building the page, refusing a webcam round with no frame and keeping a BGR copy.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=2, default_seed=None)

    # Without a frame, Generate shows the controller's no-sketch message.
    with pytest.raises(gradio.Error, match=re.escape(NO_SKETCH_MESSAGE)):
        page.run_generation("webcam", None, "chair", 2, None, 0, False, progress=lambda *args, **kwargs: None)

    # A streamed RGB frame is kept in BGR order and enables Generate.
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 255
    button = page.keep_frame(frame, "webcam", None, "chair")
    kept = page.latest_frame
    assert kept[0, 0].tolist() == [0, 0, 255], f"Expected the frame in BGR order, got {kept[0, 0].tolist()}"
    assert button.interactive is True, "Generate must be enabled once a frame and prompt exist"

    # A frame from before the stream stopped disables Generate and is refused with its own message.
    page.latest_frame_time -= ui.STALE_FRAME_INTERVALS * page.capture_interval_seconds + 1
    assert page.update_generate_button("webcam", None, "chair").interactive is False, "A stale frame must disable Generate"
    with pytest.raises(gradio.Error, match=re.escape(ui.STALE_FRAME_MESSAGE)):
        page.run_generation("webcam", None, "chair", 2, None, 0, False, progress=lambda *args, **kwargs: None)
