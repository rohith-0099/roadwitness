# DECISIONS

Every design decision and why. Newest at the bottom.

## D001 — OpenCV pin: `opencv-python-headless==5.0.0.93`
Checked PyPI on 2026-10-12: 5.0.0.93 is the only and latest 5.x release. `cv2.__version__` reports `5.0.0`.

## D002 — Environment tooling: `uv` + `.venv`, Python 3.12
`uv` is already installed and produces a reproducible lock. Runtime deps and offline-training deps are kept in separate requirement sets so PyTorch can never leak into the runtime image.

## D003 — Plate blur uses the OpenCV Zoo plate detector, not a generic text detector
`docs/03_ARCHITECTURE.md` assumed plates would be found with a text detector (DB / PP-OCR). OpenCV Zoo ships `license_plate_detection_yunet`, a dedicated plate detector. A dedicated detector is the primary path; the text detector stays as a fallback candidate, to be decided by the Phase 0.5 spike.

## D004 — Weights and datasets never enter git
`.gitignore` blocks `*.onnx`, `*.pt`, `models/weights/`, `data/`. Each model gets a download script and a model card with source URL, license, and SHA-256.

## D005 — No cleanup of user caches without approval
Disk is 96% full. The reclaimable space (npm, Docker, pip caches) is outside this repo, so it is listed under ASK ME in `PROGRESS.md` instead of being deleted.

## D006 — Segmenter: LRASPP-MobileNetV3-Large semantic segmentation, not YOLO-seg
The docs named YOLOv8/YOLO11-seg. Ultralytics code and weights are AGPL-3.0, which conflicts with the project rule (MIT-compatible only, no GPL family). torchvision's LRASPP-MobileNetV3-L is BSD-3, exports to plain ONNX, and matches PyTorch to 2e-5 on `ENGINE_NEW` at 22 ms (512x512, laptop CPU). Pothole instances come from connected contours of the mask (`findContours`), which also suits the measurement step. Per-instance confidence = mean foreground probability inside the contour. Cost: no built-in instance separation for touching potholes; accepted.

## D007 — DeepLabV3-MobileNetV3 rejected; numeric parity test is mandatory
Spike 0.2 found DeepLabV3-MobileNetV3-L loads on `ENGINE_NEW` without any error but returns wrong logits (max abs err 12.4, argmax agreement 0.50 against PyTorch; `ENGINE_CLASSIC` is correct). A first run with random weights hid this because the net's output was near-constant. Consequences: (1) every model shipped gets a test comparing OpenCV output with a stored reference on a real image; (2) spikes use pretrained weights and real inputs.

## D008 — Engine selection is explicit in code, never via environment variable
Models are loaded with `cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_NEW)`. The wheel also exposes `ENGINE_ORT`; it is never used, so the "inference only through OpenCV's own engine" claim holds by construction. The meaning of `OPENCV_FORCE_DNN_ENGINE` values differs between the 5.0.0 wheel and the `5.x` branch docs, another reason not to rely on it.

## D009 — Privacy blur = union of three detectors
Faces (YuNet), plates (LPD-YuNet) and text regions (PP-OCRv3 det). LPD-YuNet was trained on Chinese plates, so Indian-plate recall is unknown; text detection covers the gap at 67 ms. Over-blurring is the safe failure direction.

## D010 — VLM is optional in the agent design
PaliGemma2 ONNX is about 12 GB fp32 and could not be tested yet. The policy gate treats "VLM unavailable" as a first-class state: uncertain detections go to re-shoot / officer flag. The VLM is added only if its spike passes.

## D011 — Test images must have clear rights
`lena.png` was removed. Fixtures: OpenCV's own `box.png` / `box_in_scene.png` (Apache-2.0 test data) and NASA's public-domain astronaut photo (via scikit-image).
