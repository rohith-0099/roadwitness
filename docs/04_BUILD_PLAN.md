# 04 — Build plan

Deadline Oct 26, 11:59 PM PT. Feature freeze **Oct 23**. Oct 24–26: report, video, fixes. Each phase ends with its acceptance check passing — don't start the next phase on a broken one.

## Phase 0 — Risk spikes (do first, ~1 day, write findings into docs)
Each spike is a tiny script in `spikes/` + a one-paragraph result in the relevant doc.
1. **OpenCV 5 on ARM64**: `pip install opencv-python-headless==5.0.0.93` in a `python:3.12-slim` ARM64 container; `cv2.__version__` starts with `5.`. Run on Fargate ARM64 once.
2. **Segmentation model in new engine**: export YOLOv8n-seg (or YOLO11n-seg) to ONNX → `cv2.dnn.readNetFromONNX` with new engine forced (`OPENCV_FORCE_DNN_ENGINE=2`) → correct output shapes; time it.
3. **ALIKED + LightGlue**: find the Python API and weights; match two photos of the same pothole from different angles; print inlier count after USAC homography. Time it.
4. **VLM in OpenCV**: run `samples/dnn/vlm_inference.py` with PaliGemma2 ONNX; prompt a yes/no question on a pothole crop; record latency on Graviton and answer quality on 20 crops. Decide: keep, swap checkpoint, or fall back (document why).
5. **Privacy models**: YuNet face + text detector from OpenCV Zoo load and run in OpenCV 5.
6. **COOL**: find the AWS Marketplace listing, its OpenCV version, how it's delivered, cost. Decide hybrid vs skip COOL award.
**Exit:** a table in this doc: spike → works? → latency → decision.

### Phase 0 results (measured 2026-10-12, laptop x86_64 16 threads, `opencv-python-headless==5.0.0.93`; Graviton numbers pending AWS access)
| Spike | Works? | Latency (warm, CPU) | Decision |
|---|---|---|---|
| 0.1 OpenCV 5 install | Yes locally (`cv2.__version__ == 5.0.0`). ARM64 container / Fargate: pending | n/a | Pin `5.0.0.93` (only 5.x on PyPI). |
| 0.2 Segmentation on new DNN engine (`spikes/spike2_*.py`) | LRASPP-MobileNetV3-L: yes, max abs err 2e-5 vs PyTorch. FCN-ResNet50: yes, 2e-5. **DeepLabV3-MobileNetV3-L: loads but output is wrong on `ENGINE_NEW`** (max abs err 12.4, argmax agreement 0.50); correct on `ENGINE_CLASSIC`. | LRASPP 22 ms new / 35 ms classic; DeepLabV3 54 / 88 ms; FCN-R50 614 / 875 ms (512x512) | Use **LRASPP-MobileNetV3-L** semantic segmentation + contours. Not Ultralytics YOLO (AGPL-3.0, see DECISIONS D006). Every exported model must pass a numeric parity test against its PyTorch reference. |
| 0.3 ALIKED + LightGlue + USAC (`spikes/spike3_aliked_lightglue.py`) | Yes on new engine. box / box_in_scene: 1000 kpts each, 211 matches (equals OpenCV's own reference count), 207 `USAC_MAGSAC` inliers. Real same-pothole pair: pending own captures. | ALIKED ~520 ms / image, LightGlue ~430 ms / pair, USAC ~1 ms | Keep. API: `cv2.ALIKED_create`, `cv2.LightGlueMatcher_create` + `setPairInfo(kpts1, kpts2, size1, size2)` before `match`. Weights: same files OpenCV's tests use (SHA-1 verified). |
| 0.4 VLM (PaliGemma2 in `cv2.dnn`) | **Not run yet.** ONNX export is fp32, about 12 GB on disk; local machine has 4 GB free. Sample on the `5.x` branch uses `cv.dnn.ENGINE_OPENCV`, which the 5.0.0 wheel does not have (it has `ENGINE_NEW`). | unknown | Blocked on disk space. Agent design must work with the VLM disabled (gate falls through to re-shoot / officer flag). |
| 0.5 Privacy models (`spikes/spike5_privacy.py`) | YuNet face: yes. LPD-YuNet plate: yes on new engine, same boxes as classic. PP-OCRv3 text det via `TextDetectionModel_DB`: yes. | face 6 ms, plate 7 ms, text 67 ms | Blur = faces + plates + text regions (union). LPD-YuNet was trained on Chinese plates; Indian-plate recall is unmeasured, so text detection stays on as a second net and blur coverage gets a manual audit. |
| 0.6 COOL | pending | | |

## Phase 1 — Data + detector (parallel with Phase 0 tail)
- Collect own data: ≥300 frames from local roads (varied light, wet/dry), ≥30 same-pothole pairs from different angles/days, a **trap set** (see 05_EVAL).
- Public data: road-damage/pothole datasets — **check each license before use** and record it in `eval/datasets.md`.
- Fine-tune seg model offline; export ONNX; model card in `models/`.
**Exit:** detector mAP/precision/recall on held-out local set recorded (honest, whatever it is).

## Phase 2 — Vertical slice A: video → potholes on a map (local, no cloud)
`vision/quality.py`, `privacy.py`, `detect.py`, `measure.py` + tests. CLI: `roadwitness analyze video.mp4 --gps track.gpx` → JSON + annotated keyframes.
**Exit:** one real ride video produces blurred, annotated evidence images and a JSON of potholes with areas.

## Phase 3 — Slice B: dedupe + re-identification
`vision/match.py`, `index.py`. Pothole entity store (local SQLite stand-in for DynamoDB behind an interface).
**Exit:** on the same-pothole pair set, same-place decision precision/recall reported; two reports of the same pothole merge.

## Phase 4 — Slice C: agent + policy gates + cases
`agent/policy.py` (pure, exhaustive unit tests for every branch in the loop diagram), Strands agent with tools, case state machine, Decision log.
**Exit:** scripted scenario suite (≥20 scenarios: blurry upload, puddle, duplicate, new, black spot under DLP, expired DLP…) passes with expected decisions; each run emits a decision trace.

## Phase 5 — Slice D: repair verification
`vision/verify.py`: align → re-segment → diff → verdict. Scheduler hook.
**Exit:** verdict accuracy on before/after pairs reported, including CANT_TELL behavior.

## Phase 6 — AWS deployment
CDK stacks: storage, queue, worker (ARM64), API, agent, scheduler, SES, observability. One command deploy + teardown. Budget alarm.
**Exit:** upload from a phone → case visible in officer console on the deployed URL; X-Ray/CloudWatch trace shows the decisions.

## Phase 7 — Product surfaces
Citizen PWA (record/upload, re-shoot prompts with overlay, my reports), officer console (queue, evidence, approve/route, decision trace, overrides), public map (open/overdue/fixed/reopened). Accessible: large touch targets, readable contrast, works on low-end Android over 4G.
**Exit:** a non-technical friend completes a report without help (note it as a usability test in the report).

## Phase 8 — MCP server + COOL benchmark (stretch)
`roadwitness-mcp` with the tools in 03. COOL vs stock OpenCV (ARM64) vs x86 benchmark script with fixed inputs → `eval/results/cool_bench.md`.

## Phase 9 — Submission pack (Oct 24–26)
- Technical report (sections exactly as the rules list), architecture diagram (SVG + PNG), agent workflow diagram, 2–3 decision traces.
- README: one-liner, GIF, quickstart, deploy, test, pinned deps, limitations.
- Video ≤5 min: hook story (0:00–0:30) → live capture → re-shoot moment → dedupe moment → approval → repair verification → architecture → results + failure cases → team.
- Devpost writeup: casual, honest, "challenges we ran into".
- Keep the endpoint alive through Nov 9.
