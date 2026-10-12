"""Privacy blur: faces, number plates and text regions are destroyed before storage.

Three OpenCV 5 detectors run on every frame that will be persisted or shown:
YuNet (faces), LPD-YuNet (plates) and PP-OCRv3 detection (text, as a plate fallback,
because LPD-YuNet was trained on Chinese plates). Their boxes are unioned, padded and
pixelated. Over-blurring is the safe failure direction.

Plate decoding follows OpenCV Zoo's lpd_yunet.py (Apache-2.0), see docs/ATTRIBUTIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from models.registry import model_path

ENGINE = cv2.dnn.ENGINE_NEW


@dataclass(frozen=True)
class Region:
    kind: str                 # "face" | "plate" | "text"
    box: tuple[int, int, int, int]  # x, y, w, h in image pixels
    score: float

    def to_dict(self) -> dict:
        return {"kind": self.kind, "box": list(self.box), "score": round(self.score, 3)}


@dataclass(frozen=True)
class BlurResult:
    image: np.ndarray
    regions: tuple[Region, ...]

    @property
    def blur_applied(self) -> bool:
        return bool(self.regions)


def _clip_box(x, y, w, h, img_w, img_h, pad=0.0) -> tuple[int, int, int, int] | None:
    px, py = w * pad, h * pad
    x0, y0 = max(0, int(np.floor(x - px))), max(0, int(np.floor(y - py)))
    x1, y1 = min(img_w, int(np.ceil(x + w + px))), min(img_h, int(np.ceil(y + h + py)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1 - x0, y1 - y0


def pixelate(image: np.ndarray, box: tuple[int, int, int, int], blocks: int = 6) -> None:
    """Irreversibly destroys detail inside `box`, in place.

    The region is averaged down to at most `blocks` cells per side and scaled back up,
    so the original pixels cannot be recovered by deconvolution the way a Gaussian
    blur sometimes can.
    """
    x, y, w, h = box
    roi = image[y:y + h, x:x + w]
    small = cv2.resize(roi, (max(1, min(blocks, w)), max(1, min(blocks, h))), interpolation=cv2.INTER_AREA)
    image[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class FaceDetector:
    def __init__(self, score_threshold: float = 0.6, max_side: int = 960):
        self.max_side = max_side
        self._det = cv2.FaceDetectorYN_create(str(model_path("face")), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        work = image if s == 1.0 else cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        self._det.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self._det.detect(work)
        out = []
        for f in ([] if faces is None else faces):
            box = _clip_box(f[0] / s, f[1] / s, f[2] / s, f[3] / s, w, h, pad=0.15)
            if box:
                out.append(Region("face", box, float(f[-1])))
        return out


class PlateDetector:
    _MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    _STEPS = [8, 16, 32, 64]

    def __init__(self, score_threshold: float = 0.7, nms_threshold: float = 0.3, input_size=(320, 240)):
        self.score_threshold, self.nms_threshold = score_threshold, nms_threshold
        self.input_size = input_size
        self._net = cv2.dnn.readNetFromONNX(str(model_path("plate")), ENGINE)
        self._priors = self._make_priors(*input_size)

    @classmethod
    def _make_priors(cls, w: int, h: int) -> np.ndarray:
        fm = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        priors = []
        for k, step in enumerate(cls._STEPS):
            fm = [fm[0] // 2, fm[1] // 2]
            for i, j in product(range(fm[0]), range(fm[1])):
                for ms in cls._MIN_SIZES[k]:
                    priors.append([(j + 0.5) * step / w, (i + 0.5) * step / h, ms / w, ms / h])
        return np.array(priors, dtype=np.float32)

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        iw, ih = self.input_size
        self._net.setInput(cv2.dnn.blobFromImage(cv2.resize(image, (iw, ih))))
        loc, conf, iou = self._net.forward(["loc", "conf", "iou"])
        scores = np.sqrt(conf[:, 1] * np.clip(iou[:, 0], 0.0, 1.0))
        pri, scale = self._priors, np.array([iw, ih], dtype=np.float32)
        corners = np.hstack([(pri[:, 0:2] + loc[:, a:a + 2] * 0.1 * pri[:, 2:4]) * scale for a in (4, 6, 10, 12)])
        xs, ys = corners[:, 0::2] * (w / iw), corners[:, 1::2] * (h / ih)
        boxes = np.stack([xs.min(1), ys.min(1), xs.max(1) - xs.min(1), ys.max(1) - ys.min(1)], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.score_threshold, self.nms_threshold)
        out = []
        for i in np.array(keep).reshape(-1):
            box = _clip_box(*boxes[i], w, h, pad=0.2)
            if box:
                out.append(Region("plate", box, float(scores[i])))
        return out


class TextDetector:
    def __init__(self, input_size=(736, 736), max_area_fraction: float = 0.10):
        # A "text" box covering a large share of the frame is a false positive on road
        # texture; blurring it would destroy the evidence, so it is ignored.
        self.max_area_fraction = max_area_fraction
        self._model = cv2.dnn_TextDetectionModel_DB(str(model_path("text")))
        self._model.setInputParams(1.0 / 255.0, input_size, (122.67891434, 116.66876762, 104.00698793))
        self._model.setBinaryThreshold(0.3)
        self._model.setPolygonThreshold(0.5)

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        rects, confs = self._model.detectTextRectangles(image)
        out = []
        for rect, conf in zip(rects, confs):
            x, y, bw, bh = cv2.boundingRect(cv2.boxPoints(rect).astype(np.float32))
            if bw * bh > self.max_area_fraction * w * h:
                continue
            box = _clip_box(x, y, bw, bh, w, h, pad=0.1)
            if box:
                out.append(Region("text", box, float(conf)))
        return out


class PrivacyBlur:
    """Loads the detectors once; `apply` returns a blurred copy plus what was blurred."""

    def __init__(self, faces: bool = True, plates: bool = True, text: bool = True):
        self._detectors = []
        if faces:
            self._detectors.append(FaceDetector())
        if plates:
            self._detectors.append(PlateDetector())
        if text:
            self._detectors.append(TextDetector())

    def detect(self, image: np.ndarray) -> list[Region]:
        regions: list[Region] = []
        for det in self._detectors:
            regions.extend(det.detect(image))
        return regions

    def apply(self, image: np.ndarray) -> BlurResult:
        if image is None or image.size == 0:
            raise ValueError("empty image")
        regions = self.detect(image)
        out = image.copy()
        for r in regions:
            pixelate(out, r.box)
        return BlurResult(out, tuple(regions))
