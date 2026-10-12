# PROGRESS

If context resets: re-read `CLAUDE.md`, this file, and `DECISIONS.md`, then continue from **Next step**.

Last updated: 2026-10-12

## Current status
Phase 0 done except the two spikes that need things only you can provide (VLM: disk; ARM64/Fargate: emulation or AWS). Phase 2 started. Local dev env up: Python 3.12.3, `opencv-python-headless==5.0.0.93` (latest 5.x on PyPI, checked 2026-10-12), x86_64 laptop, RTX 3050 4 GB.

## Next step
Phase 2 slice A: `vision/quality.py` (done when tests pass) → `vision/privacy.py` → `vision/detect.py` → `vision/measure.py` → CLI. In parallel Phase 1: pick a license-clean public pothole segmentation dataset, write `eval/datasets.md` and `eval/SHOT_LIST.md`. Spike 0.4 (VLM) and 0.1b wait on ASK ME items.

## Phase checklist
- [ ] **Phase 0 — Risk spikes**
  - [x] 0.1a OpenCV 5 installs and imports locally (x86_64): `cv2.__version__ == "5.0.0"`
  - [ ] 0.1b OpenCV 5 in `python:3.12-slim` ARM64 container: aarch64 wheel exists on PyPI; local run blocked (no QEMU binfmt), Fargate run needs AWS creds
  - [x] 0.2 Segmentation ONNX on new DNN engine (LRASPP chosen; DeepLabV3 wrong on new engine)
  - [x] 0.3 ALIKED + LightGlue + USAC homography (real pothole pair still pending captures)
  - [ ] 0.4 PaliGemma2 VLM via `cv2.dnn` (blocked on disk space, see ASK ME)
  - [x] 0.5 Privacy models (YuNet face, LPD-YuNet plate, PP-OCRv3 text)
  - [x] 0.6 COOL: researched (AMI, OpenCV 5.0 based, ~$0.01/h software on c8g.large); run pending AWS
- [ ] Phase 1 — Data + detector
- [ ] Phase 2 — Slice A: video → potholes on a map (local)
- [ ] Phase 3 — Slice B: dedupe + re-identification
- [ ] Phase 4 — Slice C: agent + policy gates + cases
- [ ] Phase 5 — Slice D: repair verification
- [ ] Phase 6 — AWS deployment
- [ ] Phase 7 — Product surfaces
- [ ] Phase 8 — MCP server + COOL benchmark (stretch)
- [ ] Phase 9 — Submission pack

## Verified so far (API facts, OpenCV 5.0.0 Python wheel)
- `cv2.dnn.readNetFromONNX(path, engine)`; `cv2.dnn.ENGINE_AUTO=3`, `ENGINE_CLASSIC=1`, `ENGINE_NEW=2`, `ENGINE_ORT=4`.
- `cv2.ALIKED_create(modelPath, params)`; `cv2.ALIKED_Params` has `inputSize`, `normalizeDescriptors`, `engine`, `backend`, `target`.
- `cv2.LightGlueMatcher_create(modelPath, scoreThreshold, backend, target)`; has `setPairInfo` / `clearPairInfo` and the usual `match` / `knnMatch`.
- `cv2.ANNIndex_create` exists (Annoy binding), with `ANNINDEX_DIST_ANGULAR|EUCLIDEAN|DOTPRODUCT|MANHATTAN|HAMMING`.
- `cv2.FaceDetectorYN_create`, `cv2.dnn_TextDetectionModel_DB`, `cv2.dnn.Tokenizer.load/encode/decode`, `cv2.aruco.*`, `cv2.USAC_MAGSAC` all present.
- OpenCV Zoo on Hugging Face has `opencv/license_plate_detection_yunet` (dedicated plate detector) and `opencv/face_detection_yunet`.

## Blockers
- **Disk**: 4.1 GB free on `/` (98% full) after adding the offline export env (`.venv-train`, 0.9 GB). PaliGemma2 ONNX (several GB) and offline PyTorch training env will not fit.
- **No AWS access**: `aws` CLI not installed, no credentials. Phases 0.1b (Fargate run) and 6 wait on this.

## ASK ME
1. **Free disk space** (needed for VLM spike and training). Reclaimable outside this project, not touched without your OK:
   - `~/.npm` cache: 15 GB (`npm cache clean --force`)
   - Docker build cache: 1.9 GB (`docker builder prune`)
   - `~/.cache/pip`: 1.4 GB (`pip cache purge`)
2. **AWS**: account + credentials (`aws configure` or SSO), Bedrock model access, one SES verified sender address. No AWS resource has been created; a cost list will be added here before any is.
3. **ARM64 emulation** (optional, lets me build and test the ARM64 image locally): `docker run --privileged --rm tonistiigi/binfmt --install arm64`. It registers QEMU with the host kernel, so I have not run it myself.
4. **Real captures** (exact shot list will be written to `eval/SHOT_LIST.md`).
