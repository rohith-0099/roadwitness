"""Download registered model weights and verify their SHA-256.

    python -m models.download            # all models
    python -m models.download face plate # a subset
    python -m models.download --check    # verify what is on disk, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

from models.registry import MODELS, WEIGHTS_DIR, ModelSpec


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: ModelSpec) -> str:
    """Returns 'ok', 'downloaded' or raises on hash mismatch."""
    if spec.path.exists() and sha256_of(spec.path) == spec.sha256:
        return "ok"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_suffix(spec.path.suffix + ".part")
    urllib.request.urlretrieve(spec.url, tmp)
    got = sha256_of(tmp)
    if got != spec.sha256:
        tmp.unlink()
        raise RuntimeError(f"{spec.name}: sha256 mismatch, expected {spec.sha256}, got {got}")
    tmp.replace(spec.path)
    return "downloaded"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"models to fetch (default: all of {', '.join(MODELS)})")
    ap.add_argument("--check", action="store_true", help="only verify files already on disk")
    args = ap.parse_args(argv)
    unknown = [n for n in args.names if n not in MODELS]
    if unknown:
        ap.error(f"unknown model(s): {', '.join(unknown)}")
    failed = False
    for name in args.names or MODELS:
        spec = MODELS[name]
        try:
            if args.check:
                if not spec.path.exists():
                    status = "missing"
                else:
                    status = "ok" if sha256_of(spec.path) == spec.sha256 else "HASH MISMATCH"
                failed |= status != "ok"
            else:
                status = fetch(spec)
        except Exception as e:  # network or hash failure: report and keep going
            status, failed = f"FAILED: {e}", True
        print(f"{name:10s} {spec.filename:55s} {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
