"""Spike 0.2 (runtime half): run exported segmentation ONNX in OpenCV 5's DNN engines.

Usage: python spikes/spike2_seg_engine.py <dir_with_onnx_and_ref_npy>
Reports, per model and engine: loads?, max abs error vs the PyTorch reference, latency.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ENGINES = {"NEW": cv2.dnn.ENGINE_NEW, "CLASSIC": cv2.dnn.ENGINE_CLASSIC}


def main(d):
    d = Path(d)
    x = np.load(d / "input.npy")
    print("cv2", cv2.__version__, "| threads", cv2.getNumThreads(), "| input", x.shape)
    for onnx in sorted(d.glob("*.onnx")):
        ref = np.load(d / f"{onnx.stem}.ref.npy")
        for ename, eng in ENGINES.items():
            try:
                net = cv2.dnn.readNetFromONNX(str(onnx), eng)
                ts = []
                for _ in range(6):
                    t = time.perf_counter()
                    net.setInput(x)
                    y = net.forward()
                    ts.append((time.perf_counter() - t) * 1e3)
                err = float(np.abs(y - ref).max())
                agree = float((y.argmax(1) == ref.argmax(1)).mean())
                print(f"{onnx.stem:18s} {ename:8s} OK  out {y.shape} max|err| {err:.2e} "
                      f"argmax agree {agree:.4f} first {ts[0]:.0f} ms warm {np.median(ts[1:]):.0f} ms")
            except cv2.error as e:
                print(f"{onnx.stem:18s} {ename:8s} FAIL {str(e).splitlines()[0][:160]}")


if __name__ == "__main__":
    main(sys.argv[1])
