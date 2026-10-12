"""Spike 0.2 (offline half): export permissively licensed segmentation nets to ONNX.

Run with the OFFLINE env only: .venv-train/bin/python spikes/spike2_export.py <out_dir>
Weights are torchvision's COCO-pretrained ones and the input is a real image, so the
numerical comparison against PyTorch is meaningful (random-init nets gave near-constant
outputs, which made the first version of this check vacuous). This spike checks operator
coverage, numerical agreement and latency in the OpenCV 5 DNN engine, not pothole accuracy.
torchvision model code is BSD-3-Clause.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models import segmentation as seg

SIZE = 512
CANDIDATES = {
    "lraspp_mnv3l": lambda: seg.lraspp_mobilenet_v3_large(weights="DEFAULT"),
    "deeplabv3_mnv3l": lambda: seg.deeplabv3_mobilenet_v3_large(weights="DEFAULT"),
    "fcn_resnet50": lambda: seg.fcn_resnet50(weights="DEFAULT"),
}


class OutOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    import cv2  # image IO only; inference here is PyTorch, as the reference
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "tests/fixtures/astronaut.png"))
    img = cv2.cvtColor(cv2.resize(img, (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(img.transpose(2, 0, 1)[None].copy())
    np.save(out / "input.npy", x.numpy())
    for name, build in CANDIDATES.items():
        model = OutOnly(build()).eval()
        with torch.no_grad():
            y = model(x)
        np.save(out / f"{name}.ref.npy", y.numpy())
        for opset in (17,):
            path = out / f"{name}.onnx"
            torch.onnx.export(model, x, str(path), opset_version=opset, dynamo=False,
                              input_names=["image"], output_names=["logits"])
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name}: {params:.1f}M params, out {tuple(y.shape)}, "
                  f"{path.stat().st_size / 1e6:.1f} MB, opset {opset}")


if __name__ == "__main__":
    main(sys.argv[1])
