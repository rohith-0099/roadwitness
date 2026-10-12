"""OFFLINE: fine-tune LRASPP-MobileNetV3-L for road-surface segmentation and export ONNX.

    .venv-train/bin/python -m models.training.train_seg --epochs 40
    .venv-train/bin/python -m models.training.train_seg --export-only

PyTorch is used here and only here. The product runs the exported ONNX through
cv2.dnn. Output channels (sigmoid, independent): 0 pothole, 1 crack, 2 road.
The exported graph takes RGB in 0..1 (N,3,H,W), normalises inside the graph and ends
in a sigmoid, so the runtime needs no model-specific arithmetic.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import segmentation as seg

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "public" / "cprid" / "v1"
MANIFEST = ROOT / "data" / "prepared" / "cprid_manifest.json"
RUN = ROOT / "runs" / "seg"
CHANNELS = ("POTHOLE", "CRACK", "LANE")
CROP = 512
EXPORT_HW = (640, 1024)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_sample(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = RAW / sample_id
    img = cv2.cvtColor(cv2.imread(str(d / f"{sample_id}_RAW.jpg")), cv2.COLOR_BGR2RGB)
    masks = [cv2.imread(str(d / f"{sample_id}_{c}.png"), cv2.IMREAD_GRAYSCALE) for c in CHANNELS]
    return img, (np.stack(masks, -1) > 127).astype(np.uint8)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random scale, pothole-biased crop, flip, brightness / contrast / blur."""
    s = rng.uniform(0.75, 1.4)
    h, w = img.shape[:2]
    nh, nw = max(CROP, round(h * s)), max(CROP, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask[..., 0])
    if len(ys) and rng.random() < 0.8:  # centre the crop near a pothole pixel
        k = rng.integers(len(ys))
        y0 = int(np.clip(ys[k] - rng.integers(CROP // 4, 3 * CROP // 4), 0, nh - CROP))
        x0 = int(np.clip(xs[k] - rng.integers(CROP // 4, 3 * CROP // 4), 0, nw - CROP))
    else:
        y0, x0 = int(rng.integers(0, nh - CROP + 1)), int(rng.integers(0, nw - CROP + 1))
    img, mask = img[y0:y0 + CROP, x0:x0 + CROP], mask[y0:y0 + CROP, x0:x0 + CROP]
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img = img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(mask)


class Net(torch.nn.Module):
    """LRASPP with in-graph normalisation and sigmoid, so the ONNX is self-contained."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.m = seg.lraspp_mobilenet_v3_large(weights="DEFAULT" if pretrained else None,
                                               weights_backbone=None)
        head = self.m.classifier
        head.low_classifier = torch.nn.Conv2d(head.low_classifier.in_channels, len(CHANNELS), 1)
        head.high_classifier = torch.nn.Conv2d(head.high_classifier.in_channels, len(CHANNELS), 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def logits(self, x):
        return self.m((x - self.mean) / self.std)["out"]

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


def loss_fn(logits, target):
    weight = torch.tensor([4.0, 1.0, 1.0], device=logits.device).view(1, 3, 1, 1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
    p = torch.sigmoid(logits)
    inter = (p * target).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
    return bce + (dice * weight.view(3) / weight.sum()).sum()


@torch.no_grad()
def evaluate(model, items, limit=None) -> dict:
    """Pixel IoU per channel over a split, full frames, threshold 0.5."""
    model.eval()
    inter, union = np.zeros(3), np.zeros(3)
    for it in items[:limit]:
        img, mask = load_sample(it["id"])
        pred = (model(to_tensor(img)[None])[0].numpy() > 0.5)
        gt = mask.transpose(2, 0, 1).astype(bool)
        inter += (pred & gt).sum((1, 2))
        union += (pred | gt).sum((1, 2))
    iou = inter / np.maximum(union, 1)
    return {"iou_pothole": float(iou[0]), "iou_crack": float(iou[1]), "iou_road": float(iou[2])}


def export(model, path: Path) -> None:
    model.eval()
    x = torch.rand(1, 3, *EXPORT_HW)
    torch.onnx.export(model, x, str(path), opset_version=17, dynamo=False,
                      input_names=["image"], output_names=["probs"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    splits = json.loads(MANIFEST.read_text())["splits"]
    model = Net()
    best_path = RUN / "best.pt"
    if args.export_only:
        model.load_state_dict(torch.load(best_path))
        export(model, RUN / "roadseg_lraspp.onnx")
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pos = [i for i in splits["train"] if i["pothole_fraction"] > 0]
    neg = [i for i in splits["train"] if i["pothole_fraction"] == 0]
    steps = args.epochs * (2 * len(pos) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    log = []
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        # Every epoch: all pothole frames plus an equal number of random negatives.
        order = pos + [neg[i] for i in rng.choice(len(neg), len(pos), replace=False)]
        rng.shuffle(order)
        t0, total = time.time(), 0.0
        nb = len(order) // args.batch
        for b in range(nb):
            xs, ys = [], []
            for it in order[b * args.batch:(b + 1) * args.batch]:
                img, mask = augment(*load_sample(it["id"]), rng)
                xs.append(to_tensor(img))
                ys.append(torch.from_numpy(mask.transpose(2, 0, 1).copy()).float())
            loss = loss_fn(model.logits(torch.stack(xs)), torch.stack(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val = evaluate(model, splits["val"])
        row = {"epoch": epoch + 1, "loss": round(total / nb, 4), "sec": round(time.time() - t0),
               **{k: round(v, 4) for k, v in val.items()}}
        log.append(row)
        print(json.dumps(row), flush=True)
        if val["iou_pothole"] > best:  # model selection on the validation highways only
            best = val["iou_pothole"]
            torch.save(model.state_dict(), best_path)
        (RUN / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
    model.load_state_dict(torch.load(best_path))
    export(model, RUN / "roadseg_lraspp.onnx")
    print("exported", RUN / "roadseg_lraspp.onnx", "best val iou_pothole", round(best, 4))


if __name__ == "__main__":
    main()
