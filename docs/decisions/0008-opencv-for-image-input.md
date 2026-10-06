# ADR 0008: OpenCV for image input and preprocessing

Status: accepted

## Context

T02a needs to decode PNG and JPEG sketches and preprocess them (grayscale, crop to the drawing, contrast, resize), and T02b needs webcam capture (R01). The standard-library PNG parser from T01b could not decode pixels or read JPEG. OpenCV was already the candidate toolkit in `docs/ARCHITECTURE.md` and `docs/REFERENCES.md`.

## Decision

Add `opencv-python-headless==5.0.0.93` (Apache 2.0) as a runtime dependency for decoding, preprocessing, and later webcam capture. The headless build is used because the app shows images through the terminal and later the UI, not OpenCV windows. Declare `numpy>=2` directly because the code imports it. numpy is not pinned exactly because the newest release requires Python 3.12, while the project supports 3.11. Pillow was the alternative for decoding, but it lacks perspective correction and camera capture, so it would be a second dependency later.

## Consequences

Default installs are heavier (OpenCV plus numpy). Images are decoded from bytes with `cv2.imdecode`, so non-ASCII paths work on Windows. The installed versions on the development machine were OpenCV 5.0.0 and numpy 2.5.3 on Python 3.12. A lock or constraints file is still pending and should be added before the first real-model dependency.
