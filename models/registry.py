"""Registry of every model the runtime loads: where it comes from, its license, its hash.

Weights are never committed. `python -m models.download` fetches them into
`models/weights/` and refuses any file whose SHA-256 differs from the pin below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WEIGHTS_DIR = Path(os.environ.get("ROADWITNESS_WEIGHTS", Path(__file__).resolve().parent / "weights"))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    license: str
    source: str
    used_for: str

    @property
    def path(self) -> Path:
        return WEIGHTS_DIR / self.filename


_HF = "https://huggingface.co/opencv/{repo}/resolve/main/{file}"
_LG = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/{file}"

MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec(
        name="aliked",
        filename="aliked-n16rot-top1k-640.onnx",
        url=_LG.format(file="aliked-n16rot-top1k-640.onnx"),
        sha256="b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea",
        license="BSD-3-Clause (ALIKED, Shiaoming/ALIKED)",
        source="ONNX export referenced by opencv_extra testdata/dnn/download_models.py",
        used_for="keypoints + descriptors for re-identification and before/after alignment",
    ),
    ModelSpec(
        name="lightglue",
        filename="aliked_lightglue.onnx",
        url=_LG.format(file="aliked_lightglue.onnx"),
        sha256="a1bd99d294c4c6c08ca978af3128c85f8c0e603b04a03e4391428455d9624439",
        license="Apache-2.0 (LightGlue, cvg/LightGlue)",
        source="ONNX export referenced by opencv_extra testdata/dnn/download_models.py",
        used_for="feature matching for re-identification and before/after alignment",
    ),
    ModelSpec(
        name="face",
        filename="face_detection_yunet_2023mar.onnx",
        url=_HF.format(repo="face_detection_yunet", file="face_detection_yunet_2023mar.onnx"),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT (OpenCV Zoo, YuNet)",
        source="huggingface.co/opencv/face_detection_yunet",
        used_for="privacy blur: faces",
    ),
    ModelSpec(
        name="plate",
        filename="license_plate_detection_lpd_yunet_2023mar.onnx",
        url=_HF.format(repo="license_plate_detection_yunet",
                       file="license_plate_detection_lpd_yunet_2023mar.onnx"),
        sha256="6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7",
        license="Apache-2.0 (OpenCV Zoo, LPD-YuNet)",
        source="huggingface.co/opencv/license_plate_detection_yunet",
        used_for="privacy blur: number plates",
    ),
    ModelSpec(
        name="text",
        filename="text_detection_en_ppocrv3_2023may.onnx",
        url=_HF.format(repo="text_detection_ppocr", file="text_detection_en_ppocrv3_2023may.onnx"),
        sha256="03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
        license="Apache-2.0 (OpenCV Zoo, PP-OCRv3 detection)",
        source="huggingface.co/opencv/text_detection_ppocr",
        used_for="privacy blur: text regions (plate fallback)",
    ),
]}


def model_path(name: str) -> Path:
    """Path of a registered model; raises with the fix if it has not been downloaded."""
    spec = MODELS[name]
    if not spec.path.exists():
        raise FileNotFoundError(
            f"model '{name}' not found at {spec.path}; run `python -m models.download {name}`")
    return spec.path


def available(*names: str) -> bool:
    return all(MODELS[n].path.exists() for n in names)
