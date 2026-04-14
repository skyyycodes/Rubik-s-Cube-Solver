#!/usr/bin/env python3
"""
Held-out eval of the shipped neural move predictors.

Generates a fresh, uniformly-depth-sampled, shuffled validation set with a
fixed RNG seed, then scores every pickle in kociemba/kociemba/neural_model*.pkl
on that same set. Reports top-1 / top-3 overall plus top-1 broken down by
scramble-depth bucket so metric artefacts (e.g. depth-1 contamination) are
visible.

Label scheme: inverse of last scramble move (the same proxy used by the
paper's training code). This is deliberately the friendly label -- if a
model can't hit high accuracy on its own training objective on held-out
data, it is not a useful move-orderer.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "kociemba"))

from kociemba.pykociemba.coordcube import CoordCube
from kociemba.pykociemba.cubiecube import CubieCube, moveCube
from kociemba.neural_heuristic import extract_coordinate_features

N_MOVES = 18
N_EVAL = 10_000
MAX_DEPTH = 25
SEED = 12345

N_TWIST, N_FLIP, N_SLICE = 2187, 2048, 495
N_PARITY, N_URFDLF, N_FRTOBR = 2, 20160, 11880


def scalar_features(twist, flip, slice_coord, parity, URFtoDLF, FRtoBR):
    return np.array([
        twist / N_TWIST,
        flip / N_FLIP,
        slice_coord / N_SLICE,
        parity / N_PARITY,
        URFtoDLF / N_URFDLF,
        FRtoBR / N_FRTOBR,
        (twist % 729) / 729,
        (flip % 256) / 256,
        (slice_coord % 99) / 99,
    ], dtype=np.float32)


def apply_move(cube: CubieCube, mv: int) -> None:
    axis, power = mv // 3, mv % 3
    for _ in range(power + 1):
        cube.cornerMultiply(moveCube[axis])
        cube.edgeMultiply(moveCube[axis])


def generate_eval_set(n: int, seed: int):
    rng = np.random.default_rng(seed)
    depths = rng.integers(1, MAX_DEPTH + 1, size=n)

    feats_coord = np.empty((n, 161), dtype=np.float32)
    feats_scalar = np.empty((n, 9), dtype=np.float32)
    labels = np.empty(n, dtype=np.int32)

    for i, depth in enumerate(depths):
        cube = CubieCube()
        last_axis = -1
        last_move = -1
        for _ in range(int(depth)):
            while True:
                mv = int(rng.integers(0, N_MOVES))
                if mv // 3 != last_axis:
                    break
            apply_move(cube, mv)
            last_axis = mv // 3
            last_move = mv

        coord = CoordCube(cube)
        slice_c = coord.FRtoBR // 24
        feats_coord[i] = extract_coordinate_features(
            coord.twist, coord.flip, slice_c,
            coord.parity, coord.URFtoDLF, coord.FRtoBR,
        )
        feats_scalar[i] = scalar_features(
            coord.twist, coord.flip, slice_c,
            coord.parity, coord.URFtoDLF, coord.FRtoBR,
        )

        axis, power = last_move // 3, last_move % 3
        labels[i] = axis * 3 + (2 - power)

    return depths, feats_coord, feats_scalar, labels


def forward(x: np.ndarray, weights, biases) -> np.ndarray:
    for w, b in zip(weights[:-1], biases[:-1]):
        x = np.maximum(0, x @ w + b)
    return x @ weights[-1] + biases[-1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def score(pickle_path: Path, feats_coord, feats_scalar, labels, depths):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
    weights = model["weights"]
    biases = model["biases"]
    input_dim = weights[0].shape[0]

    if input_dim == 161:
        feats, fmode = feats_coord, "coordinate"
    elif input_dim == 9:
        feats, fmode = feats_scalar, "scalar"
    else:
        raise ValueError(f"unexpected input dim {input_dim} in {pickle_path}")

    logits = forward(feats, weights, biases)
    top1 = np.argmax(logits, axis=1)
    # top-3 via argpartition then check membership
    top3_idx = np.argpartition(-logits, 3, axis=1)[:, :3]
    top1_hit = (top1 == labels)
    top3_hit = np.any(top3_idx == labels[:, None], axis=1)

    overall_top1 = float(top1_hit.mean())
    overall_top3 = float(top3_hit.mean())

    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
    per_bucket = []
    for lo, hi in buckets:
        mask = (depths >= lo) & (depths <= hi)
        if mask.any():
            per_bucket.append((lo, hi, int(mask.sum()),
                               float(top1_hit[mask].mean())))

    return {
        "path": str(pickle_path.relative_to(REPO)),
        "sha256": sha256(pickle_path),
        "feature_mode_file": model.get("feature_mode", "(not set)"),
        "feature_mode_inferred": fmode,
        "input_dim": input_dim,
        "hidden_sizes": model.get("hidden_sizes", "?"),
        "size_bytes": pickle_path.stat().st_size,
        "top1": overall_top1,
        "top3": overall_top3,
        "per_bucket": per_bucket,
    }


def fmt_row(r: dict) -> str:
    out = []
    out.append(f"  file         : {r['path']}")
    out.append(f"  sha256       : {r['sha256'][:16]}...")
    out.append(f"  size         : {r['size_bytes']:,} bytes")
    out.append(f"  feature_mode : file={r['feature_mode_file']!r}  "
               f"inferred={r['feature_mode_inferred']!r}  input_dim={r['input_dim']}")
    out.append(f"  hidden_sizes : {r['hidden_sizes']}")
    out.append(f"  top-1        : {r['top1']*100:6.2f}%")
    out.append(f"  top-3        : {r['top3']*100:6.2f}%")
    out.append(f"  per-depth top-1:")
    for lo, hi, n, acc in r["per_bucket"]:
        out.append(f"      depth [{lo:2d},{hi:2d}]  n={n:5d}  "
                   f"top-1={acc*100:6.2f}%")
    return "\n".join(out)


def main():
    print("=" * 72)
    print("Held-out neural move predictor eval")
    print("=" * 72)
    print(f"Eval seed       : {SEED}")
    print(f"Eval set size   : {N_EVAL:,}")
    print(f"Depth distrib.  : uniform integer in [1, {MAX_DEPTH}]")
    print(f"Label scheme    : inverse of last scramble move (proxy label)")
    print(f"Random baseline : {100/N_MOVES:.2f}%")
    print()
    print(f"Generating {N_EVAL:,} held-out cubes...")
    depths, feats_coord, feats_scalar, labels = generate_eval_set(N_EVAL, SEED)

    depth_hist = np.bincount(depths, minlength=MAX_DEPTH + 1)[1:]
    print(f"Depth histogram : {dict(zip(range(1, MAX_DEPTH+1), depth_hist.tolist()))}")
    print()

    model_dir = REPO / "kociemba" / "kociemba"
    candidates = sorted(model_dir.glob("neural_model*.pkl"))
    # exclude value predictor and .bak files
    candidates = [p for p in candidates
                  if "value" not in p.name and p.suffix == ".pkl"]

    for p in candidates:
        print("-" * 72)
        try:
            r = score(p, feats_coord, feats_scalar, labels, depths)
            print(fmt_row(r))
        except Exception as e:
            print(f"  {p.name}: FAILED -- {e}")
        print()


if __name__ == "__main__":
    main()
