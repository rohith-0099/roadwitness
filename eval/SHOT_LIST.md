# Shot list (what only you can capture)

Phone held landscape, 1080p, 30 fps, location ON (keep the GPS track: use the PWA once deployed, or any GPX logger app until then). Do not film people's faces deliberately. Copy files to `data/local/<set>/` with the folder names below.

## A. Ride videos — `data/local/rides/` (detector + dedupe data)
- **12 videos, 20–40 s each**, pillion or dashboard mount, 15–30 km/h.
- At least **6 different roads** (the test split is by road). Name files `<road>_<date>_<n>.mp4`.
- Mix: 6 dry daylight, 2 wet road, 2 overcast or dusk, 2 with strong sun / shadows from trees.
- Ride the **same 3 roads twice** on different days (this gives the dedupe ratio).

## B. Same-pothole pairs — `data/local/reid/` (re-identification)
- **30 potholes**, each photographed **4 times**: two angles (about 30° apart) on day 1, and two on a different day. Use a second phone for some if you can.
- Distance 2–4 m, pothole filling roughly a quarter of the frame, some surrounding road visible (edges, markings, kerb are what the matcher uses).
- Name `p<id>_d<day>_<a|b>.jpg`. Two potholes within about 10 m of each other count as hard negatives; include **10 such neighbours**.

## C. Repair pairs — `data/local/repair/` (verification)
- As many as real life allows; even 5 is useful and will be reported as n=5.
- For each: a "before" photo (from set B is fine) and an "after" photo from a similar position once it is patched. Also photograph **5 that were not repaired** after 1–2 weeks, and any **badly patched** one (sunken, cracked or partial patch).
- Name `r<id>_before.jpg`, `r<id>_after.jpg` and add a line to `labels.csv`: `id,verdict` with verdict one of `FIXED,NOT_FIXED,POOR_PATCH`.

## D. Trap set — `data/local/traps/` (false-positive test)
**5 photos of each**: puddle on intact road, tree or pole shadow, manhole cover, speed breaker, patched asphalt, cracks without a pothole, wet road, night / street-light, motion-blurred frame, sun glare.

## E. Measurement — `data/local/aruco/`
- Print the ArUco card (`docs/aruco_card.pdf`, generated in Phase 2) at 100% scale.
- **8 potholes**: card flat on the road beside the pothole, photo from standing height; then measure the longest length and the width across with a tape and add `id,length_cm,width_cm` to `measurements.csv`.

## F. Privacy audit — `data/local/privacy/`
- **20 street frames** that contain vehicles with plates and, where it happens naturally, pedestrians at a distance. These are used only to count blur coverage and are deleted afterwards.
