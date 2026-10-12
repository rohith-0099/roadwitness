# 02 — Rules, judging, deliverables

Source: competition page + rules overview (opencv26.devpost.com, opencv.org/opencv-ai-competition-2026). **Devpost Official Rules override this overview if they conflict — re-check them before submitting** (especially: whether pre-existing code may be reused; write RoadWitness code fresh, reuse ideas only).

## Pass/fail requirements
- [ ] **OpenCV 5** used for **substantive** image/video analysis (not just resize/IO).
- [ ] A **meaningful component runs on AWS**.
- [ ] Submitted by **Oct 26, 2026, 11:59 PM PT** (= Oct 27, ~12:29 PM IST).
- [ ] Safe, lawful, appropriate for public demo. No data/models/media without rights or consent. No privacy/surveillance risk without safeguards. **No misrepresentation of capabilities, results, benchmarks, or the role of human review.**

## Final submission must include
- [ ] **Technical report**: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible-use considerations.
- [ ] **Code repo/archive** accessible to judges (can be private). 
- [ ] **Pinned dependencies** + clear build, deploy, and test instructions.
- [ ] **Architecture diagram** showing OpenCV 5 + AWS components (+ COOL / agent components).
- [ ] **Working web endpoint** (or arranged live screen-share demo). Keep it up through Nov 9 (judging).
- [ ] **Video ≤ 5 min** (public/unlisted): shows the team, the app working, architecture, principal results.
- [ ] **Evaluation evidence**, including failure cases and limitations.

## Overall rubric (100)
| Criterion | Weight | What wins it for us |
|---|---|---|
| Technical execution | **30%** | Depth of OpenCV 5: new DNN engine, in-OpenCV VLM, ALIKED/LightGlue, USAC, TRUCO contours, privacy models. Reliability + real eval numbers. |
| Innovation | 20% | Accountability loop (verify the repair), re-identification across phones/days, warranty routing. |
| Real-world impact | 20% | Death statistics, existing legal mechanism, pilot on real roads, evidence of benefit (dedupe ratio, verified repairs). |
| User experience | 10% | No-form capture, specific re-shoot guidance, officer one-click approve, accessible UI. |
| Docs & presentation | 10% | Report, README, diagram, ≤5-min video with a story. |
| Cloud, reproducibility, responsible ops | 10% | CDK one-command deploy, ARM64 containers, CloudWatch/X-Ray traces, IAM least-privilege, privacy blur, retention policy. |

Ties break on **Technical execution**, then **Real-world impact**.

## Agentic Vision Award ($1,000) — we are going for it
Qualifies only if **image/video results influence a subsequent plan, tool call, action, or request for human approval**. A chatbot explaining a fixed vision result does not count. Using an AI coding assistant to build the entry does not count as agentic.

Rubric: OpenCV 5 + agent integration 30% · orchestration & appropriate autonomy 25% · task effectiveness & evaluation 20% · failure handling, observability, security, human control 15% · UX/docs/demo 10%.

Required evidence:
- [ ] Agent workflow diagram: perception → decision/orchestration → action.
- [ ] A **trace** showing OpenCV 5 output changing a later decision/tool call/action (we log these; show 2–3 in report + video).
- [ ] Evaluation of task success, failure handling, observability, human control.

## Best Use of COOL Award ($1,000) — stretch
COOL must execute the **claimed core workload on AWS Graviton** (or the Arm part of a documented hybrid). Needs: COOL version + instance config, reproducible method, baseline comparison (latency/throughput/cost), evidence COOL actually ran. Rubric: verified COOL integration 30% · architecture 25% · measured value 20% · innovation 15% · reproducibility 10%.

## Prize mechanics
One overall award max per entry, plus one or both special awards. At least two independent judges per entry, averaged. Judges named on page: Phil Nelson (OpenCV), Gary Bradski (OpenCV founder). AWS may nominate up to two.

## IP note
We keep ownership. Submitted *materials* (report, video, proposal) get a perpetual royalty-free license to OpenCV and AWS. Code/weights are not covered unless included in the submission.
