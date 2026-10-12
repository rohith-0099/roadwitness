"""Spike 0.5: privacy models (YuNet face, LPD-YuNet plate, PP-OCR text det) on OpenCV 5.

Usage: python spikes/spike5_privacy.py [face_image] [plate_image]
Plate decode follows OpenCV Zoo's lpd_yunet.py (Apache-2.0).
"""
import sys
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "models" / "weights"


def timed(fn, n=5):
    out, ts = None, []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return out, ts[0], float(np.median(ts[1:]))


def face(img):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN_create(str(W / "face_detection_yunet_2023mar.onnx"), "", (w, h), 0.6, 0.3, 5000)
    (_, faces), first, warm = timed(lambda: det.detect(img))
    n = 0 if faces is None else len(faces)
    print(f"[face] YuNet: {n} face(s), first {first:.0f} ms, warm {warm:.0f} ms")
    if n:
        print("       box+score:", np.round(faces[0][:4]).astype(int).tolist(), round(float(faces[0][-1]), 3))


def lpd_priors(w, h):
    min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    fm = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
    maps = []
    for _ in steps:
        fm = [fm[0] // 2, fm[1] // 2]
        maps.append(fm)
    pri = []
    for k, f in enumerate(maps):
        for i, j in product(range(f[0]), range(f[1])):
            for ms in min_sizes[k]:
                pri.append([(j + 0.5) * steps[k] / w, (i + 0.5) * steps[k] / h, ms / w, ms / h])
    return np.array(pri, dtype=np.float32)


def plate(img, engine):
    w, h = 320, 240
    net = cv2.dnn.readNetFromONNX(str(W / "license_plate_detection_lpd_yunet_2023mar.onnx"), engine)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (w, h)))
    pri = lpd_priors(w, h)

    def run():
        net.setInput(blob)
        return net.forward(["loc", "conf", "iou"])

    (loc, conf, iou), first, warm = timed(run)
    scores = np.sqrt(conf[:, 1] * np.clip(iou[:, 0], 0, 1))
    scale = np.array([w, h], dtype=np.float32)
    corners = np.hstack([(pri[:, 0:2] + loc[:, a:a + 2] * 0.1 * pri[:, 2:4]) * scale for a in (4, 6, 10, 12)])
    xs, ys = corners[:, 0::2], corners[:, 1::2]
    boxes = np.stack([xs.min(1), ys.min(1), xs.max(1) - xs.min(1), ys.max(1) - ys.min(1)], 1)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.8, 0.3)
    print(f"[plate] LPD-YuNet engine={engine}: outputs {loc.shape} {conf.shape} {iou.shape}, "
          f"{len(keep)} plate(s), first {first:.0f} ms, warm {warm:.0f} ms")
    for i in list(keep)[:3]:
        print("        box(320x240)+score:", np.round(boxes[i]).astype(int).tolist(), round(float(scores[i]), 3))


def text(img):
    m = cv2.dnn_TextDetectionModel_DB(str(W / "text_detection_en_ppocrv3_2023may.onnx"))
    m.setInputParams(1.0 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793))
    m.setBinaryThreshold(0.3)
    m.setPolygonThreshold(0.5)
    (rects, confs), first, warm = timed(lambda: m.detectTextRectangles(img), n=3)
    print(f"[text] PP-OCRv3 det: {len(rects)} region(s), first {first:.0f} ms, warm {warm:.0f} ms")


if __name__ == "__main__":
    print("cv2", cv2.__version__)
    face_img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tests/fixtures/astronaut.png"))
    face(face_img)
    plate_img = cv2.imread(sys.argv[2]) if len(sys.argv) > 2 else None
    if plate_img is not None:
        for eng in (cv2.dnn.ENGINE_NEW, cv2.dnn.ENGINE_CLASSIC):
            try:
                plate(plate_img, eng)
            except cv2.error as e:
                print(f"[plate] engine={eng} FAILED: {str(e).splitlines()[0][:200]}")
        try:
            text(plate_img)
        except cv2.error as e:
            print(f"[text] FAILED: {str(e).splitlines()[0][:200]}")
