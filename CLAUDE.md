# RoadWitness — Claude Code project brief

> "Roads come with a warranty nobody enforces. RoadWitness turns a phone video into evidence that enforces it."

RoadWitness is an **agentic vision product**: a citizen records a short road video, and an agent built on **OpenCV 5** finds potholes, verifies them, measures them, merges duplicates, routes the case to whoever is legally responsible (often the contractor, during the road's defect-liability / warranty period), and later **visually proves whether the repair actually happened**.

Built for the **OpenCV AI Competition 2026 (OpenCV 5 + AWS)**. Target: 1st place overall + Agentic Vision Award (+ Best Use of COOL if feasible). It must read as a **product**, not a hackathon demo.

## Read these first (in order)
1. `docs/01_PRODUCT.md` — problem, users, product surface, story
2. `docs/02_RULES_AND_JUDGING.md` — hard rules, rubric, deliverables. **Pass/fail rules live here.**
3. `docs/03_ARCHITECTURE.md` — OpenCV 5 feature map, agent loop, AWS architecture, data model, MCP tools
4. `docs/04_BUILD_PLAN.md` — risk spikes first, then vertical slices with acceptance criteria
5. `docs/05_EVAL.md` — how we measure ourselves (this is scored)

## Non-negotiables
- **OpenCV 5 is the brain, not a resize helper.** All inference (detector, segmenter, VLM, features) runs through OpenCV 5 (`cv2.dnn`, Features module, geometry/USAC). Training may happen outside OpenCV; **inference never does**. No PyTorch / onnxruntime in the runtime image.
- **Vision output must change what the agent does next** (re-shoot request, VLM second opinion, merge vs new case, route, close vs reopen). A chatbot narrating a vision result is explicitly disqualified from the Agentic award.
- **LLM proposes, Python enforces.** The LLM can plan and draft; every irreversible action passes deterministic gates (thresholds, geometry checks) and a **human approval** step. Never send anything to an authority/contractor without approval.
- **Privacy before persistence.** Faces and number plates are blurred (OpenCV) before anything is stored in a shared/public bucket or shown on a map.
- **Honesty.** Never fabricate metrics, registry data, or results. Demo data is labeled as demo data. Failure cases are documented, not hidden.
- **Real AWS.** Core vision workload runs on AWS (ECS Fargate ARM64/Graviton). Not a token Lambda.

## Working style
- Build in **vertical slices** (capture → detect → case → map) that run end-to-end before deepening any layer.
- Every OpenCV capability gets a small pure-Python module with **unit tests** (fixtures in `tests/fixtures/`). Cloud-free tests must pass with `pytest -q`.
- Keep business logic in plain tested Python; frameworks (Strands, FastAPI, Next.js) are thin envelopes.
- Pin every dependency (`requirements.lock`, `package-lock.json`). Python 3.12. `opencv-python-headless==5.0.0.93` (verify it's the latest 5.x when starting; ARM64 wheels exist).
- Anything marked **VERIFY** in the docs is an assumption about a new API — check it in a 10-minute spike before building on it, and update the doc with what you found.
- Log every agent decision with its visual evidence (trace id, frame id, scores) — judges need a trace showing OpenCV output changing a decision.

## Repo layout (target)
```
roadwitness/
  vision/            # pure OpenCV 5 modules (quality, privacy, detect, measure, match, verify, vlm)
  agent/             # policy gates (pure Python) + Strands agent + tools
  mcp_server/        # roadwitness-mcp (FastMCP) exposing vision tools
  api/               # FastAPI: uploads, cases, approvals, webhooks
  worker/            # SQS consumer running the vision pipeline (ARM64 container)
  web/               # Next.js: citizen PWA, officer console, public map
  infra/             # AWS CDK (TypeScript or Python)
  models/            # model cards + download scripts (weights not committed)
  eval/              # datasets manifest, labeling guide, eval scripts, results/
  docs/
  tests/
```

## Deadline
Final submission: **Oct 26, 2026, 11:59 PM Pacific** (= Oct 27, ~12:29 PM IST). Feature freeze **Oct 23**; Oct 24–26 are for report, video, and fixes only.
