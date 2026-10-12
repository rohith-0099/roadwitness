"""Frame quality gate and keyframe selection (classic OpenCV imgproc, no models).

The gate decides whether a frame is worth analysing. A failing frame produces a
specific reason, which the agent turns into a targeted re-shoot request instead of
running the detector on unusable pixels.

Thresholds are provisional defaults. They get calibrated on labelled local captures
(eval/quality_gate.py) and the calibrated values are recorded in eval/results/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Iterator

import cv2
import numpy as np

ANALYSIS_WIDTH = 640  # metrics are computed at a fixed width so they compare across phones


class Reason(str, Enum):
    BLUR = "blur"
    DARK = "dark"
    OVEREXPOSED = "overexposed"
    GLARE = "glare"


RESHOOT_HINT = {
    Reason.BLUR: "The video is blurry. Hold the phone steady or slow down and record again.",
    Reason.DARK: "The road is too dark to see. Record again in daylight or under a street light.",
    Reason.OVEREXPOSED: "The image is washed out. Point the camera away from the sun and record again.",
    Reason.GLARE: "Strong reflection hides the road surface. Change the angle so the glare moves off the pothole.",
}


@dataclass(frozen=True)
class QualityThresholds:
    min_sharpness: float = 60.0       # variance of Laplacian at ANALYSIS_WIDTH
    min_brightness: float = 45.0      # mean luma, 0..255
    max_dark_fraction: float = 0.60   # share of pixels with luma < 30
    max_bright_fraction: float = 0.35  # share of pixels with luma > 245
    max_glare_fraction: float = 0.12  # share of pixels that are bright and colourless


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reasons: tuple[Reason, ...]
    sharpness: float
    brightness: float
    dark_fraction: float
    bright_fraction: float
    glare_fraction: float
    hints: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = [r.value for r in self.reasons]
        d["hints"] = list(self.hints)
        return d


def _analysis_gray(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bgr, gray) resized to ANALYSIS_WIDTH."""
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    if w != ANALYSIS_WIDTH:
        frame = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, round(h * ANALYSIS_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian at analysis width. Higher is sharper."""
    _, gray = _analysis_gray(frame)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_frame(frame: np.ndarray, thresholds: QualityThresholds = QualityThresholds()) -> QualityReport:
    bgr, gray = _analysis_gray(frame)
    n = gray.size
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    brightness = float(np.dot(hist, np.arange(256)) / n)
    dark_fraction = float(hist[:30].sum() / n)
    bright_fraction = float(hist[246:].sum() / n)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    glare = cv2.inRange(hsv, (0, 0, 240), (180, 40, 255))
    glare_fraction = float(cv2.countNonZero(glare) / n)

    t = thresholds
    reasons: list[Reason] = []
    if brightness < t.min_brightness or dark_fraction > t.max_dark_fraction:
        reasons.append(Reason.DARK)
    if bright_fraction > t.max_bright_fraction:
        reasons.append(Reason.OVEREXPOSED)
    elif glare_fraction > t.max_glare_fraction:
        reasons.append(Reason.GLARE)
    # A dark or blown-out frame has little texture, so blur is only reported when
    # exposure is fine: the citizen gets the one instruction that fixes the capture.
    if sharp < t.min_sharpness and not reasons:
        reasons.append(Reason.BLUR)

    return QualityReport(
        ok=not reasons,
        reasons=tuple(reasons),
        sharpness=round(sharp, 2),
        brightness=round(brightness, 2),
        dark_fraction=round(dark_fraction, 4),
        bright_fraction=round(bright_fraction, 4),
        glare_fraction=round(glare_fraction, 4),
        hints=tuple(RESHOOT_HINT[r] for r in reasons),
    )


@dataclass(frozen=True)
class Keyframe:
    frame_idx: int
    frame: np.ndarray
    quality: QualityReport


def iter_video(path: str, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_idx, frame) for every `stride`-th frame of a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def select_keyframes(frames: Iterable[tuple[int, np.ndarray]], window: int = 15,
                     thresholds: QualityThresholds = QualityThresholds()) -> tuple[list[Keyframe], list[QualityReport]]:
    """Picks the sharpest passing frame from each run of `window` input frames.

    Returns (keyframes, all_reports). A window with no passing frame contributes no
    keyframe; its reports stay in all_reports so the caller can explain a rejection.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    keyframes: list[Keyframe] = []
    reports: list[QualityReport] = []
    best: Keyframe | None = None
    for count, (idx, frame) in enumerate(frames, start=1):
        rep = assess_frame(frame, thresholds)
        reports.append(rep)
        if rep.ok and (best is None or rep.sharpness > best.quality.sharpness):
            best = Keyframe(idx, frame, rep)
        if count % window == 0:
            if best is not None:
                keyframes.append(best)
            best = None
    if best is not None:
        keyframes.append(best)
    return keyframes, reports


def dominant_reason(reports: list[QualityReport]) -> Reason | None:
    """Most frequent failure reason across a rejected upload, for the re-shoot message."""
    counts: dict[Reason, int] = {}
    for rep in reports:
        for r in rep.reasons:
            counts[r] = counts.get(r, 0) + 1
    return max(counts, key=counts.get) if counts else None
