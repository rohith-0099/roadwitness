# Datasets

Every dataset used or considered, with source, license, size, split, and known problems. Nothing here is committed to git; `data/` is ignored.

## 1. CPRID — "Cracks and Potholes in Road Images" (public, training + public benchmark)
| | |
|---|---|
| Source | Passos, Cassaniga, Fernandes, Medeiros, Comunello. Mendeley Data, v4, doi:[10.17632/t576ydh9v8.4](https://data.mendeley.com/datasets/t576ydh9v8/4) |
| License | **CC BY 4.0** (confirmed via the Mendeley Data API on 2026-10-12) |
| Provenance | Vehicle-camera frames of Brazilian federal highways (ES, RS, DF), 2014–2017, supplied by DNIT under Brazil's Access to Information Law, protocol 50650.003556/2017-28 |
| Content | 2,235 frames, 1024x640 or 1024x630. Three binary masks per frame: pothole, crack, road. 564 frames contain pothole pixels; pothole area is small (median 0.43% of the frame, 90th percentile 4.7%) |
| Archive SHA-256 | `e59bb8d7…ec4f76`, pinned in full in `models/training/prepare_data.py` |
| Our split | **By highway**, fixed before any training: test = RS_386 (460 frames, 125 with potholes); val = DF_080 + DF_020 (307 frames, 83 with potholes); train = the remaining 8 highways (1,468 frames, 356 with potholes) |
| Used for | Training the segmenter and a public, reproducible benchmark. |

Known limits: Brazilian highways, forward-facing vehicle camera, daylight. No Indian roads, no pillion or handheld views, no night or rain. The domain gap to our own captures is unmeasured until the local set exists.

## 2. Local set (own captures) — not collected yet
See `eval/SHOT_LIST.md`. Headline detector, re-ID and repair metrics must come from this set. Split by road, not by frame.

## Considered and rejected
| Dataset | Stated license | Why rejected |
|---|---|---|
| `manot/pothole-segmentation` (Roboflow Universe export, Hugging Face) | CC BY 4.0 | Inspected 2026-10-12 and deleted. (1) Contains Alamy-watermarked stock photos and TV-news screenshots, so the uploader's license cannot cover the images. (2) Its 819 files are only 269 distinct photos, and the published splits leak: 63 of 72 test sources also appear in train (37 of 40 checked pairs pixel-identical). |
| Indian Roads semantic segmentation (e-Yantra, IIT Bombay) | GPL 2.0 | GPL-family license. |
| RDD2022 | CC BY-SA 4.0 | Bounding boxes only, no masks; share-alike terms. |
| Ultralytics pretrained weights | AGPL-3.0 | See `DECISIONS.md` D006. |
