"""OFFLINE: download CPRID and write a split manifest, held out by highway.

    .venv-train/bin/python -m models.training.prepare_data

Dataset: "Cracks and Potholes in Road Images" (Passos et al., Mendeley Data,
doi:10.17632/t576ydh9v8.4), CC BY 4.0. 2,235 vehicle-camera frames from Brazilian
federal highways supplied by DNIT, each with POTHOLE / CRACK / LANE (road) masks.

The split is by highway, never by frame: consecutive frames of one road are near
duplicates, so a frame-level split would leak. Output: data/prepared/cprid_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "public" / "cprid"
MANIFEST = ROOT / "data" / "prepared" / "cprid_manifest.json"
URL = ("https://data.mendeley.com/public-files/datasets/t576ydh9v8/files/"
       "afc7c028-06e0-475b-b190-e008df681b19/file_downloaded")
SHA256 = "e59bb8d771e18f059ab78d88e640e83869fc8c8fd808697a0c2c48e16eec4f76"

# Whole highways are held out. Chosen once, before any training, by size only.
TEST_HIGHWAYS = {"RS_386"}
VAL_HIGHWAYS = {"DF_080", "DF_020"}


def highway_of(sample: str) -> str:
    """'1092813_DF_251_251BDF0052_00615' -> 'DF_251' (state + highway number)."""
    parts = sample.split("_")
    return f"{parts[1]}_{parts[2]}"


def split_of(sample: str) -> str:
    hw = highway_of(sample)
    return "test" if hw in TEST_HIGHWAYS else "val" if hw in VAL_HIGHWAYS else "train"


def download() -> None:
    if (RAW / "v1").is_dir():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    z = RAW / "cprid.zip"
    urllib.request.urlretrieve(URL, z)
    h = hashlib.sha256()
    with open(z, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise RuntimeError(f"cprid.zip sha256 mismatch: {h.hexdigest()}")
    with zipfile.ZipFile(z) as f:
        f.extractall(RAW)
    z.unlink()


def main() -> None:
    download()
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for d in sorted((RAW / "v1").iterdir()):
        mask = cv2.imread(str(d / f"{d.name}_POTHOLE.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or not (d / f"{d.name}_RAW.jpg").exists():
            continue
        splits[split_of(d.name)].append({
            "id": d.name, "highway": highway_of(d.name),
            "pothole_fraction": round(float((mask > 127).mean()), 6)})
    summary = {}
    for name, items in splits.items():
        summary[name] = {"images": len(items),
                         "with_pothole": sum(i["pothole_fraction"] > 0 for i in items),
                         "highways": dict(Counter(i["highway"] for i in items))}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "dataset": "Cracks and Potholes in Road Images (CPRID)", "doi": "10.17632/t576ydh9v8.4",
        "license": "CC BY 4.0", "split_by": "highway", "summary": summary, "splits": splits}, indent=1) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
