from pathlib import Path

import cv2
import numpy as np
import pytest

from models.registry import available
from vision.privacy import PlateDetector, PrivacyBlur, Region, _clip_box, pixelate

FIXTURES = Path(__file__).parent / "fixtures"
needs_models = pytest.mark.skipif(not available("face", "plate", "text"),
                                  reason="weights missing: run `python -m models.download`")


def road_texture(seed=0):
    return np.random.default_rng(seed).normal(110, 25, (480, 640, 3)).clip(0, 255).astype(np.uint8)


def synthetic_plate():
    img = road_texture()
    cv2.rectangle(img, (220, 300), (420, 350), (255, 255, 255), -1)
    cv2.rectangle(img, (220, 300), (420, 350), (0, 0, 0), 2)
    cv2.putText(img, "KL 07 AB 1234", (228, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# --- pure helpers (no models) -------------------------------------------------

def test_clip_box_pads_and_clips():
    assert _clip_box(10, 10, 20, 10, 100, 100, pad=0.5) == (0, 5, 40, 20)
    assert _clip_box(90, 90, 30, 30, 100, 100) == (90, 90, 10, 10)
    assert _clip_box(200, 200, 10, 10, 100, 100) is None  # fully outside


def test_pixelate_destroys_detail_only_inside_box():
    img = road_texture()
    before = img.copy()
    box = (100, 120, 80, 40)
    pixelate(img, box, blocks=4)
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    assert len(np.unique(roi.reshape(-1, 3), axis=0)) <= 16
    mask = np.ones(img.shape[:2], bool)
    mask[y:y + h, x:x + w] = False
    assert np.array_equal(img[mask], before[mask])


def test_pixelate_handles_tiny_box():
    img = road_texture()
    pixelate(img, (5, 5, 2, 2))
    assert len(np.unique(img[5:7, 5:7].reshape(-1, 3), axis=0)) <= 4


def test_plate_priors_match_model_output_count():
    # 320x240 is the size the model was published with: 4385 anchors.
    assert PlateDetector._make_priors(320, 240).shape == (4385, 4)


def test_region_to_dict():
    assert Region("face", (1, 2, 3, 4), 0.98765).to_dict() == {"kind": "face", "box": [1, 2, 3, 4], "score": 0.988}


# --- with models ----------------------------------------------------------------

@pytest.fixture(scope="module")
def blur():
    return PrivacyBlur()


@needs_models
def test_face_is_blurred_and_input_untouched(blur):
    img = cv2.imread(str(FIXTURES / "astronaut.png"))
    original = img.copy()
    res = blur.apply(img)
    assert np.array_equal(img, original), "apply must not modify its input"
    faces = [r for r in res.regions if r.kind == "face"]
    assert len(faces) == 1 and res.blur_applied
    x, y, w, h = faces[0].box
    # The detected box must cover the face centre (hand-checked location in this photo).
    assert x < 223 < x + w and y < 120 < y + h
    assert not np.array_equal(res.image[y:y + h, x:x + w], original[y:y + h, x:x + w])
    # A second pass must not find the face again: the blur is strong enough.
    assert not [r for r in blur.detect(res.image) if r.kind == "face"]


@needs_models
def test_plate_text_is_caught_by_text_fallback(blur):
    img = synthetic_plate()
    res = blur.apply(img)
    assert res.blur_applied
    covered = np.zeros(img.shape[:2], bool)
    for r in res.regions:
        x, y, w, h = r.box
        covered[y:y + h, x:x + w] = True
    # The characters span roughly x 228..410, y 312..338.
    assert covered[315:336, 232:405].mean() > 0.95


@needs_models
def test_plain_road_is_not_blurred(blur):
    img = road_texture(3)
    res = blur.apply(img)
    assert res.regions == () and not res.blur_applied
    assert np.array_equal(res.image, img)


@needs_models
def test_empty_image_raises(blur):
    with pytest.raises(ValueError):
        blur.apply(np.zeros((0, 0, 3), np.uint8))


@needs_models
def test_detectors_can_be_disabled():
    only_faces = PrivacyBlur(plates=False, text=False)
    assert only_faces.apply(synthetic_plate()).regions == ()
