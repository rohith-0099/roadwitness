"""Spike 0.3: ALIKED + LightGlue + USAC homography through OpenCV 5 only.

Usage: python spikes/spike3_aliked_lightglue.py [img1 img2]
Defaults to OpenCV's own box / box_in_scene test pair.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "models" / "weights"
ALIKED = str(W / "aliked-n16rot-top1k-640.onnx")
LIGHTGLUE = str(W / "aliked_lightglue.onnx")


def kpts_to_mat(kpts):
    return np.array([k.pt for k in kpts], dtype=np.float32).reshape(-1, 2)


def main():
    p1, p2 = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (
        str(ROOT / "tests/fixtures/box.png"), str(ROOT / "tests/fixtures/box_in_scene.png"))
    img1, img2 = cv2.imread(p1), cv2.imread(p2)
    assert img1 is not None and img2 is not None, "could not read images"
    print("cv2", cv2.__version__, "| img1", img1.shape, "| img2", img2.shape)

    params = cv2.ALIKED_Params()
    print("ALIKED engine param:", params.engine, "(ENGINE_NEW =", cv2.dnn.ENGINE_NEW, ")")
    t = time.perf_counter()
    detector = cv2.ALIKED_create(ALIKED, params)
    matcher = cv2.LightGlueMatcher_create(LIGHTGLUE, 0.0)
    print(f"load: {(time.perf_counter() - t) * 1e3:.0f} ms")

    times = {}
    for run in range(3):
        t = time.perf_counter()
        k1, d1 = detector.detectAndCompute(img1, None)
        k2, d2 = detector.detectAndCompute(img2, None)
        times.setdefault("aliked_x2", []).append(time.perf_counter() - t)

        t = time.perf_counter()
        m1, m2 = kpts_to_mat(k1), kpts_to_mat(k2)
        matcher.setPairInfo(m1, m2, (img1.shape[1], img1.shape[0]), (img2.shape[1], img2.shape[0]))
        matches = matcher.match(d1, d2)
        times.setdefault("lightglue", []).append(time.perf_counter() - t)

        t = time.perf_counter()
        src = np.float32([m1[m.queryIdx] for m in matches])
        dst = np.float32([m2[m.trainIdx] for m in matches])
        H, mask = (None, None)
        if len(matches) >= 4:
            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
        times.setdefault("usac", []).append(time.perf_counter() - t)

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"keypoints: {len(k1)} / {len(k2)} | descriptors: {d1.shape} {d1.dtype}")
    print(f"matches: {len(matches)} | USAC_MAGSAC inliers: {inliers} "
          f"({inliers / max(len(matches), 1):.2f})")
    print("H:\n", None if H is None else np.round(H, 3))
    for k, v in times.items():
        print(f"{k}: first {v[0] * 1e3:.0f} ms, warm median {np.median(v[1:]) * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
