# ADR 0008: OpenCV for image input and preprocessing

Status: accepted, amended in T02b (GUI build instead of headless)

## Context

T02a needs to decode PNG and JPEG sketches and preprocess them with grayscale, crop to the drawing, contrast, and resize, and T02b needs webcam capture (R01). The standard-library PNG parser from T01b could not decode pixels or read JPEG. OpenCV was already the candidate toolkit in `docs/ARCHITECTURE.md` and `docs/REFERENCES.md`.

## Decision

Add `opencv-python-headless==5.0.0.93` (Apache 2.0) as a runtime dependency for decoding, preprocessing, and later webcam capture. The headless build was chosen at first because the app showed images only through the terminal. T02b replaced it with `opencv-python==5.0.0.93`, the build with HighGUI, because `sketchloop --camera` shows a live preview window (`cv2.imshow`) until Space captures. Only one of the two builds may be installed, so the headless package must be uninstalled first. Declare `numpy>=2` directly because the code imports it. numpy is not pinned exactly because the newest release requires Python 3.12, while the project supports 3.11. Pillow was the alternative for decoding, but it lacks perspective correction and camera capture, so it would be a second dependency later.

## Consequences

Default installs are heavier (OpenCV plus numpy). On Linux the GUI build needs system GUI libraries such as `libGL`, which headless servers may lack, so CI installs them. On Apple Silicon the wheel requires macOS 13 or newer. macOS cameras open through AVFoundation, the terminal app needs camera permission, and window-close detection is unreliable there, so only Esc cancels the preview. CI runs on Ubuntu and on macOS (Apple Silicon). Images are decoded from bytes with `cv2.imdecode`, so non-ASCII paths work on Windows. The installed versions on the development machine were OpenCV 5.0.0 and numpy 2.5.3 on Python 3.12. A lock or constraints file is still pending and should be added before the first real-model dependency.
