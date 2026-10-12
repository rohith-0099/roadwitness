# 01 — Product

## The problem (real, global, measurable)
- India: pothole-related crashes killed **9,438 people in 2020–2024**, rising from 1,555 (2020) to 2,385 (2024), a **53% increase** — about 5 deaths a day (MoRTH data tabled in Parliament). [1][2]
- Responsibility is split across agencies (NHAI, state PWDs, municipalities), so accountability diffuses when someone dies. [2]
- Potholes are universal (US, UK, EU, Africa…). Judges have hit them. No explanation needed.

**The insight:** detection is not the bottleneck — reporting apps and helplines already exist. The bottleneck is **accountability**:
1. Reports are unverified (photo of a puddle? a shadow?).
2. The same pothole is reported 50 times; nobody knows they're the same.
3. Reports go to the wrong owner.
4. Cases get "closed" without a repair, and nobody checks.

**The legal lever already exists.** Road contracts worldwide include a **defect liability / warranty period** during which the contractor must repair defects at their own cost (e.g., FIDIC's Defects Notification Period, warranty clauses in US DOT contracts). Kerala (India) made it public:
- DLP boards with contractor + officer phone numbers at both ends of **2,150+ public works**; public can complain; if the contractor/officer doesn't act, escalate to the Minister's office toll-free line. Maintenance periods range from **6 months to 5 years**. [3]
- Potholes on accident **black spots** during DLP → contractor can be **blacklisted for one year** and fined **twice the repair cost**. [4]
- Roads past DLP are covered by "running contracts" for maintenance, with numbers also displayed. [5]

Today the complaint is a phone call. No evidence, no tracking, no proof of repair.

## The product
**RoadWitness turns a phone video into evidence that enforces the road's warranty.**

### Users and what each gets
| User | Pain | RoadWitness gives |
|---|---|---|
| Citizen / rider | Reporting is effortful and pointless | Record 10–30 s of video (pillion/dashcam). No form. Live status of "their" pothole. |
| Road engineer / officer | Flood of duplicate, low-quality complaints | Deduplicated queue ranked by **measured severity**, black-spot flags, evidence packs, one-click approve/route. |
| Contractor (under DLP) | Disputes ("that's not our stretch", "it was fine") | Timestamped, geotagged, measured evidence with a hash; clear deadline. |
| Public / journalists | No accountability | Public map: open, overdue, fixed (verified), reopened. Faces/plates blurred. |

### Core loop (what the demo shows)
1. **Capture** — video upload from a PWA. Quality gate rejects blurry/dark frames and asks for a better angle *with a specific reason*.
2. **Detect & verify** — OpenCV 5 segmentation; uncertain cases get a second opinion from a VLM running **inside OpenCV 5**; still uncertain → targeted re-shoot request.
3. **Measure** — area from the mask; real-world size when a scale reference (ArUco card) is present; relative depth/severity as stretch.
4. **Dedupe / re-identify** — ALIKED + LightGlue + USAC homography recognize the same pothole from a different phone, angle, day.
5. **Route** — look up the road segment: under DLP → contractor + engineer; black spot → urgent + cite the blacklisting clause; else → maintenance division.
6. **Act (with approval)** — agent drafts the complaint + evidence pack → **human approves** → sent.
7. **Verify repair** — at the deadline, agent requests a fresh capture; aligns before/after; decides FIXED / NOT FIXED / POOR PATCH / CAN'T TELL (→ re-shoot). Reopens if damage returns within the warranty period. Escalates if overdue.

### Hooks (use in README, video, Devpost)
- One-liner: *"Every pothole report, verified by the road itself."*
- Story opener: *"5 people die in pothole crashes in India every day. The law already makes contractors fix many of these for free. RoadWitness makes sure they do."*
- Global framing: *"Roads worldwide come with a warranty nobody enforces."* Kerala is the **pilot**, not the pitch. Explain "DLP" once, then say "warranty period".

### What makes it a product (judges should say "this is a product")
- Three real surfaces: citizen PWA, officer console, public map.
- Case lifecycle with SLAs, reminders, escalation ladder.
- Evidence pack export (PDF/HTML) with image hashes and capture metadata.
- Multi-tenant config: a city/state plugs in its own road-segment registry (CSV/GeoJSON import) and escalation ladder.
- `roadwitness-mcp`: other agents/IDEs can call `assess_road_image`, `match_pothole`, `verify_repair`.
- Observability: every agent decision is traceable to the frame and score that caused it.

### Explicit non-goals (say so in the report)
- Not surveillance: we don't track vehicles or people; faces/plates are blurred before storage.
- Not an automatic legal accusation: it produces **evidence for human review**.
- Registry in the demo is a **demo dataset modeled on real DLP board formats**, clearly labeled.

## Sources
[1] The Logical Indian — 9,438 pothole deaths 2020–2024 (MoRTH via Parliament): https://thelogicalindian.com/over-9438-lives-lost-in-india-to-pothole-accidents-between-2020-and-2024-uttar-pradesh-reports-5127-deaths/
[2] Legacy IAS — 53% rise, diffused accountability: https://www.legacyias.com/pothole-related-road-fatalities-jumped-53-in-5-years/
[3] Kerala Kaumudi — DLP boards, 2,150+ works, escalation path: https://keralakaumudi.com/web-news/en/2022/02/NMAN0309811/1.html
[4] Onmanorama — blacklisting for potholes on black spots during DLP: https://onmanorama.com/news/kerala/2024/09/03/potholes-black-spots-roads-contractors-black-listed-fined-kerala.html
[5] Kerala Kaumudi — running contracts after DLP: https://keralakaumudi.com/web-news/en/2023/02/NMAN0393457/1.html
