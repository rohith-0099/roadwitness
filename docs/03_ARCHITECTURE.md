# 03 — Architecture

## 1. OpenCV 5 feature map (the "max OpenCV" plan)
Every row is a real product function, not decoration. Facts from the OpenCV 5 release notes (github.com/opencv/opencv/wiki/OpenCV-5); items marked **VERIFY** need a spike.

| OpenCV 5 capability | Where RoadWitness uses it | Module |
|---|---|---|
| **New DNN engine** (`readNetFromONNX(..., engine)`; `OPENCV_FORCE_DNN_ENGINE=2` forces new engine; >64% ONNX coverage; YOLOv8 CPU speed on par with ONNX Runtime) | Pothole **segmentation**: LRASPP-MobileNetV3-L (torchvision, BSD-3) fine-tuned offline, exported to ONNX; instances come from contours of the mask. **VERIFIED 2026-10-12**: exact vs PyTorch on `ENGINE_NEW`, 22 ms at 512x512. YOLOv8/YOLO11 dropped: Ultralytics is AGPL-3.0. DeepLabV3-MobileNetV3 dropped: silently wrong output on `ENGINE_NEW`. All runtime inference via `cv2.dnn`. | `vision/detect.py` |
| **VLM support** — tokenizer (`cv2.dnn.Tokenizer.load`), PaliGemma2 sample `samples/dnn/vlm_inference.py` (SigLIP + embedding + Gemma2 as 3 ONNX nets) | **Second opinion** on uncertain detections only (cost-aware): VQA "is there a pothole in the marked region?" / "detect pothole". **VERIFY**: latency on Graviton; sample uses `-pt-224` checkpoint (meant for fine-tuning) and no KV-cache — test answer quality; if weak, export `paligemma2-3b-mix-224` to ONNX. Cap `max_new_tokens` (yes/no needs ~3). | `vision/vlm.py` |
| **Features module: ALIKED + LightGlue** (`cv2.LightGlueMatcher`, `cv2.detail.LightGlueFeaturesMatcher` in docs) | **Re-identification**: same pothole from different phone/angle/day → dedupe. **Repair verification**: align before/after. **VERIFIED 2026-10-12**: `cv2.ALIKED_create(model, cv2.ALIKED_Params())`, `cv2.LightGlueMatcher_create(model, scoreThreshold)`, then `matcher.setPairInfo(kpts1_Nx2, kpts2_Nx2, (w1,h1), (w2,h2))` and `matcher.match(desc1, desc2)`. Weights: the files listed in `opencv_extra/testdata/dnn/download_models.py` (`aliked-n16rot-top1k-640.onnx`, `aliked_lightglue.onnx`). | `vision/match.py` |
| **Annoy ANN search** (Features module, replaces FLANN) | Fast candidate retrieval: global descriptor per pothole crop → top-k candidates before expensive LightGlue. Scales dedupe to thousands of reports. **VERIFIED**: `cv2.ANNIndex_create(dim, cv2.ANNINDEX_DIST_ANGULAR)`, `addItems`, `build`, `knnSearch`, `save`/`load`. | `vision/index.py` |
| **USAC as default robust estimator** (MAGSAC++, PROSAC, SPRT) in `geometry` module | Homography from LightGlue matches; inlier ratio = "same place?" confidence. Road-plane fitting for depth (stretch). | `vision/match.py`, `vision/measure.py` |
| **Accelerated `warpPerspective`/`remap`** | Warp "after" image onto "before" for pixel-aligned repair comparison. | `vision/verify.py` |
| **TRUCO `findContours`** (several× faster, automatic) | Mask → polygon → area, perimeter, bbox, convexity; also for frame-quality heuristics. | `vision/measure.py` |
| **ArUco / ChArUco** (objdetect) | **Measurement mode**: officer/volunteer drops a printed ArUco card next to the pothole → real cm² area, scale for depth. | `vision/measure.py` |
| **`calib` module / multi-view calibration** | One-time phone intrinsics (ChArUco) for engineer measurement mode (stretch). | `vision/calib.py` |
| **`stereo` / `ptcloud` (TSDF, ICP, .ply export)** | Stretch: two video frames → triangulated points → road-plane fit (USAC) → **pothole depth**; export `.ply` in evidence pack. | `vision/depth3d.py` |
| **YuNet face detector** (`cv2.FaceDetectorYN`, OpenCV Zoo) + **text detection** (DB/PP-OCR det in OpenCV Zoo) | **Privacy blur**: faces + any text-like plate region blurred *before* storage. **VERIFIED 2026-10-12**: YuNet face, `license_plate_detection_yunet` (dedicated plate detector, used in addition to text detection) and PP-OCRv3 text det all run in OpenCV 5. | `vision/privacy.py` |
| **New text renderer** (STB TrueType, Unicode) | Evidence-pack overlays (measurements, IDs, timestamps). Note: complex scripts (Malayalam/Devanagari) not shaped correctly yet — keep overlay text English. | `vision/evidence.py` |
| Classic imgproc | Frame quality gate: Laplacian variance (blur), histogram (exposure), glare ratio, motion; keyframe selection from video. | `vision/quality.py` |
| **COOL on Graviton** | Preprocessing hot path (decode → resize → color/threshold → contours) on ARM64. Benchmark vs stock OpenCV ARM64 and vs x86. **Researched 2026-10-12**: COOL is an AWS Marketplace AMI (Ubuntu 24.04, Graviton4 c8g/m8g/r8g), listed as built on OpenCV 5.0, so no 4.x hybrid is needed. It is AMI-only, so it runs on EC2 next to the Fargate worker. **VERIFY** on the instance: the COOL build exposes `cv2.dnn` and Features in Python. | `worker/preprocess_cool.py` |

Rule: models are trained offline (PyTorch/Ultralytics is fine in a training notebook), exported to ONNX, and **only ever executed by OpenCV 5** in the product.

## 2. Agent loop (perception → decision → action)
Deterministic **policy gates** (pure Python, unit-tested) wrap an LLM agent (Strands Agents SDK; model via Amazon Bedrock). The LLM chooses/plans and drafts text; gates decide what is allowed.

```
upload ─► quality_gate ─fail─► REQUEST_RESHOOT(reason: blur|dark|glare|too_far)
              │ok
              ▼
        privacy_blur ─► detect_segment
              │
     conf≥τ_hi │ τ_lo≤conf<τ_hi           conf<τ_lo
        ▼      ▼                            ▼
     ACCEPT  vlm_second_opinion ──disagree──► REQUEST_RESHOOT(angle/closer, show overlay)
              │agree                         (2nd failure ► DISCARD + log)
              ▼
        measure (area; cm² if ArUco; severity class)
              ▼
        retrieve_candidates (GPS radius + ANN) ─► lightglue_match + USAC
              │ inliers≥N & homography sane ► MERGE as new observation (severity trend)
              │ ambiguous ► keep separate, flag "possible duplicate" for officer
              │ none ► NEW pothole
              ▼
        route (segment registry): DLP active ► contractor+engineer; black spot ► URGENT
              ▼
        draft complaint + evidence pack ─► HUMAN APPROVAL ─► send (SES) ─► schedule verify at deadline
              ▼ (EventBridge Scheduler fires)
        request verification capture ─► verify_repair (align, re-segment, diff)
              │ FIXED ► close (verified)   │ NOT_FIXED/POOR_PATCH ► escalate next rung (approval)
              │ CANT_TELL ► REQUEST_RESHOOT with guidance
              │ damage returns within warranty ► REOPEN
```

Where vision changes the next step (call these out in the trace/video): quality score → re-shoot; detector confidence → VLM call; VLM verdict → accept vs re-shoot; LightGlue inliers → merge vs new case; segment geometry/area → severity → priority; post-repair diff → close vs escalate vs reopen.

Autonomy levels (document in report): automatic = analysis, dedupe, drafting, scheduling. **Human-approved** = anything sent outside the system, case closure disputes, escalation. Officer can override any gate; overrides are logged and fed into eval.

## 3. AWS architecture
- **Web**: Next.js (citizen PWA + officer console + public map, Leaflet/OSM) on Amplify Hosting or S3+CloudFront.
- **Auth**: Cognito (officer accounts; citizens can be anonymous with device token).
- **API**: FastAPI on ECS Fargate (ARM64). Presigned S3 uploads.
- **Queue**: SQS (vision jobs) → **Vision worker** on ECS Fargate **ARM64/Graviton** with `opencv-python-headless==5.0.0.93` (ARM64 wheel) + models baked or pulled from S3. Optional EC2 Graviton (c7g) worker for COOL + benchmarks.
- **Agent**: Strands agent service (ECS) calling Bedrock; tools = our vision/case functions (same code exposed via MCP).
- **State**: DynamoDB (reports, observations, potholes, cases, segments, decisions). S3 (raw → private bucket with lifecycle expiry; blurred evidence → evidence bucket).
- **Time**: EventBridge Scheduler for deadline-triggered verification/escalation.
- **Messaging**: SES (sandbox → only verified addresses; demo sends to our own inboxes — say so).
- **Observability**: CloudWatch logs/metrics, X-Ray/OpenTelemetry traces; every decision row has `trace_id`. Officer console has a **Decision Trace** view (frame thumbnail + scores + gate + action).
- **IaC**: AWS CDK, one-command deploy; least-privilege IAM; secrets in Secrets Manager; budget alarm.

## 4. Data model (DynamoDB, single-table or a few tables)
- `Report` {report_id, device_id, uploaded_at, gps_track, video_key, status}
- `Observation` {obs_id, report_id, frame_idx, pothole_id?, mask_poly, area_px, area_cm2?, conf, vlm_verdict?, quality, blur_applied, image_sha256}
- `Pothole` {pothole_id, geohash, first_seen, last_seen, severity, severity_trend, observations[], segment_id, status}
- `Case` {case_id, pothole_id, owner(contractor|division), state, due_at, ladder_level, approvals[], sent[], verification[]}
- `Segment` {segment_id, geometry, road_name, agency, dlp_end?, contractor{name, phone, email}?, black_spot:bool, source:"demo"|"official"}
- `Decision` {decision_id, trace_id, case_id?, input_refs, gate, scores, outcome, actor(agent|human), ts}

Case states: `NEW → VERIFIED → ROUTED → AWAITING_APPROVAL → SENT → DUE → VERIFYING → CLOSED_FIXED | ESCALATED | REOPENED`.

## 5. MCP server (`roadwitness-mcp`, FastMCP)
Tools (thin wrappers over `vision/` and `agent/`):
- `assess_road_image(image) -> {quality, detections[], blurred_image}`
- `second_opinion(image, region) -> {verdict, rationale_tokens}`
- `measure_pothole(image, mask, aruco?) -> {area_px, area_cm2?, severity}`
- `match_pothole(image, lat, lon) -> {candidates[], best_match?, inliers, homography_ok}`
- `verify_repair(before_obs_id, after_image) -> {verdict: FIXED|NOT_FIXED|POOR_PATCH|CANT_TELL, diff_overlay}`
- `case_status(case_id)`
The agent itself calls these tools; external agents/IDEs can too ("consumer and producer").
