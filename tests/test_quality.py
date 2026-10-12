import cv2
import numpy as np
import pytest

from vision.quality import (QualityThresholds, Reason, assess_frame, dominant_reason, iter_video,
                            select_keyframes, sharpness)


def textured(seed=0, size=(480, 640)):
    """A mid-grey, asphalt-like frame with fine texture and a few edges."""
    rng = np.random.default_rng(seed)
    img = rng.normal(120, 28, (*size, 3)).clip(0, 255).astype(np.uint8)
    cv2.ellipse(img, (320, 260), (90, 50), 0, 0, 360, (40, 40, 40), -1)
    cv2.line(img, (0, 400), (640, 380), (220, 220, 220), 6)
    return img


def blurred(img, k=31):
    return cv2.GaussianBlur(img, (k, k), 0)


def test_sharp_frame_passes():
    rep = assess_frame(textured())
    assert rep.ok and rep.reasons == () and rep.hints == ()


def test_blur_is_rejected_with_hint():
    rep = assess_frame(blurred(textured()))
    assert not rep.ok
    assert rep.reasons == (Reason.BLUR,)
    assert "blurry" in rep.hints[0]


def test_dark_frame_reports_dark_not_blur():
    rep = assess_frame((textured() * 0.12).astype(np.uint8))
    assert rep.reasons == (Reason.DARK,)


def test_overexposed_frame():
    rep = assess_frame(np.full((480, 640, 3), 252, np.uint8))
    assert rep.reasons == (Reason.OVEREXPOSED,)


def test_glare_patch():
    img = textured()
    cv2.circle(img, (320, 240), 150, (255, 255, 255), -1)  # ~23% of the frame
    rep = assess_frame(img)
    assert rep.reasons == (Reason.GLARE,)


def test_metrics_are_resolution_independent():
    img = textured(size=(960, 1280))
    small = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    assert sharpness(img) == pytest.approx(sharpness(small), rel=1e-6)


def test_grayscale_input_and_empty_input():
    assert assess_frame(cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)).ok
    with pytest.raises(ValueError):
        assess_frame(np.zeros((0, 0, 3), np.uint8))


def test_thresholds_are_configurable():
    strict = QualityThresholds(min_sharpness=1e9)
    assert assess_frame(textured(), strict).reasons == (Reason.BLUR,)


def test_to_dict_is_json_ready():
    d = assess_frame(blurred(textured())).to_dict()
    assert d["reasons"] == ["blur"] and isinstance(d["hints"], list) and d["ok"] is False


def test_select_keyframes_picks_sharpest_per_window():
    sharp, soft, bad = textured(1), blurred(textured(2), 3), blurred(textured(3))
    frames = list(enumerate([bad, soft, sharp, bad, bad, bad, soft]))
    keyframes, reports = select_keyframes(frames, window=3)
    assert len(reports) == 7
    # window 1 -> frame 2 (sharpest), window 2 -> nothing passes, tail -> frame 6
    assert [k.frame_idx for k in keyframes] == [2, 6]
    assert all(k.quality.ok for k in keyframes)


def test_select_keyframes_rejects_bad_window():
    with pytest.raises(ValueError):
        select_keyframes([], window=0)


def test_dominant_reason():
    dark = (textured() * 0.1).astype(np.uint8)
    _, reports = select_keyframes(enumerate([dark, dark, blurred(textured())]), window=5)
    assert dominant_reason(reports) == Reason.DARK
    assert dominant_reason([assess_frame(textured())]) is None


def test_iter_video_roundtrip(tmp_path):
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    if not vw.isOpened():
        pytest.skip("no MJPG encoder in this OpenCV build")
    for i in range(6):
        vw.write(textured(i))
    vw.release()
    got = [idx for idx, _ in iter_video(path, stride=2)]
    assert got == [0, 2, 4]
    with pytest.raises(IOError):
        next(iter_video(str(tmp_path / "missing.avi")))
