#!/usr/bin/env python3
"""
Train the 161-dim coord-features move predictor with *pruning-table-greedy*
labels, end-to-end vectorised in NumPy.

Why prun-greedy labels
----------------------
The original training pipeline labelled each cube with ``inverse of the last
scramble move``. That label has an inherent noise ceiling at depth > 2:
there are many first-solving moves that are equally good, and the scramble
path is only one of them. More importantly, when measured properly on a
held-out uniform-depth set the resulting model scores ~15% top-1 (3x random)
-- nowhere near the 83% the paper claims.

Prun-greedy labels instead ask: "of the 18 successor states, which one has
the lowest Phase-1 admissible pruning heuristic
``max(Slice_Twist_Prun, Slice_Flip_Prun)``?" That label is:

  * deterministic (no RNG dependence given the cube)
  * cheap (three table gathers + two pruning lookups per move per cube)
  * aligned with the IDA* objective the paper argues for -- the move a
    perfect orderer inside Kociemba Phase 1 would pick next, given only
    the heuristic the solver itself uses
  * depth-independent -- a depth-20 cube and a depth-2 cube get equally
    sharp labels, so the training signal is not dominated by the shallow
    slice the way proxy labels are

Full pipeline
-------------
1. Generate N cubes by scrambling from solved for uniformly-sampled depth
   in [1, 25], tracking five coordinate channels (twist, flip, FRtoBR,
   URFtoDLF, parity) via the pre-computed Kociemba move tables. Fully
   vectorised.
2. For each cube, compute prun-greedy label in coordinate space.
3. Extract 161-dim binned-coordinate features (matches the deployed
   `extract_coordinate_features` feature composition).
4. Shuffle, 95/5 train/val split.
5. Train a 161 -> 256 -> 128 -> 64 -> 18 MLP with Adam + cosine LR decay,
   cross-entropy loss, batch 1024. Print val top-1 / top-3 every epoch.
6. Save weights plus a full provenance sidecar JSON (sha256, git commit,
   label scheme, seed, per-epoch metrics, wall time).

The deployed ``neural_model.pkl`` is **not** overwritten. This script writes
to ``kociemba/kociemba/neural_model_prun{N}.pkl`` plus a ``.meta.json``
sidecar. Promotion is a separate deliberate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "kociemba"))

from kociemba.pykociemba.coordcube import CoordCube  # noqa: E402

N_MOVES = 18
N_TWIST = CoordCube.N_TWIST          # 2187
N_FLIP = CoordCube.N_FLIP            # 2048
N_FRTOBR = CoordCube.N_FRtoBR        # 11880
N_URFDLF = CoordCube.N_URFtoDLF      # 20160
N_SLICE1 = CoordCube.N_SLICE1        # 495
N_PARITY = CoordCube.N_PARITY        # 2

# Binned-feature widths (must match extract_coordinate_features)
N_TWIST_BINS = 32
N_FLIP_BINS = 32
N_SLICE_BINS = 16
N_URFDLF_BINS = 32
N_FRTOBR_BINS = 32
FEATURE_DIM = (N_TWIST_BINS + N_FLIP_BINS + N_SLICE_BINS
               + N_URFDLF_BINS + N_FRTOBR_BINS + 2 + 9 + 6)
assert FEATURE_DIM == 161


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

def load_move_tables():
    """Convert the nested lists in CoordCube to contiguous int32 NumPy."""
    print("  loading move tables...", flush=True)
    twist_mv = np.asarray(CoordCube.twistMove, dtype=np.int32)
    flip_mv = np.asarray(CoordCube.flipMove, dtype=np.int32)
    frtobr_mv = np.asarray(CoordCube.FRtoBR_Move, dtype=np.int32)
    urfdlf_mv = np.asarray(CoordCube.URFtoDLF_Move, dtype=np.int32)
    parity_mv = np.asarray(CoordCube.parityMove, dtype=np.int32)
    assert twist_mv.shape == (N_TWIST, N_MOVES)
    assert flip_mv.shape == (N_FLIP, N_MOVES)
    assert frtobr_mv.shape == (N_FRTOBR, N_MOVES)
    assert urfdlf_mv.shape == (N_URFDLF, N_MOVES)
    assert parity_mv.shape == (N_PARITY, N_MOVES)
    return twist_mv, flip_mv, frtobr_mv, urfdlf_mv, parity_mv


def unpack_prun(packed_list, expected_len):
    """Unpack a half-byte-packed pruning table into a full uint8 array."""
    packed = np.asarray(packed_list, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed & 0x0f
    out[1::2] = (packed >> 4) & 0x0f
    return out[:expected_len]


def load_pruning_tables():
    print("  loading pruning tables...", flush=True)
    prun_twist = unpack_prun(CoordCube.Slice_Twist_Prun,
                             N_SLICE1 * N_TWIST)
    prun_flip = unpack_prun(CoordCube.Slice_Flip_Prun,
                            N_SLICE1 * N_FLIP)
    return prun_twist, prun_flip


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(n: int, seed: int, tables):
    """
    Generate n cubes vectorised, returning coordinate channels and depth.

    Each cube gets a uniformly sampled depth in [1, 25]. At each scramble
    step only cubes whose target depth > t are updated, enforcing the
    no-same-axis-twice constraint via rejection sampling.
    """
    twist_mv, flip_mv, frtobr_mv, urfdlf_mv, parity_mv = tables
    rng = np.random.default_rng(seed)

    max_depth = 25
    depth = rng.integers(1, max_depth + 1, size=n, dtype=np.int32)

    # State channels (start from solved, coord = 0 for all five)
    twist = np.zeros(n, dtype=np.int32)
    flip = np.zeros(n, dtype=np.int32)
    frtobr = np.zeros(n, dtype=np.int32)
    urfdlf = np.zeros(n, dtype=np.int32)
    parity = np.zeros(n, dtype=np.int32)
    last_axis = np.full(n, -1, dtype=np.int32)

    for t in range(max_depth):
        active = depth > t
        if not active.any():
            break
        # Rejection-sample a move per cube s.t. axis != last_axis
        mv = rng.integers(0, N_MOVES, size=n, dtype=np.int32)
        while True:
            bad = active & (mv // 3 == last_axis)
            if not bad.any():
                break
            mv[bad] = rng.integers(0, N_MOVES, size=int(bad.sum()),
                                   dtype=np.int32)

        # Gather new coord values; blend with old via `active` mask.
        # `np.where` avoids a boolean-indexed assignment on five arrays.
        new_twist = twist_mv[twist, mv]
        new_flip = flip_mv[flip, mv]
        new_frtobr = frtobr_mv[frtobr, mv]
        new_urfdlf = urfdlf_mv[urfdlf, mv]
        new_parity = parity_mv[parity, mv]
        new_last_axis = mv // 3

        twist = np.where(active, new_twist, twist)
        flip = np.where(active, new_flip, flip)
        frtobr = np.where(active, new_frtobr, frtobr)
        urfdlf = np.where(active, new_urfdlf, urfdlf)
        parity = np.where(active, new_parity, parity)
        last_axis = np.where(active, new_last_axis, last_axis)

        if (t + 1) % 5 == 0:
            print(f"    scramble step {t+1}/{max_depth} "
                  f"(active={int(active.sum())})", flush=True)

    return dict(twist=twist, flip=flip, frtobr=frtobr,
                urfdlf=urfdlf, parity=parity, depth=depth)


# ---------------------------------------------------------------------------
# Prun-greedy labels
# ---------------------------------------------------------------------------

def prun_greedy_labels(state, tables, prun_tables):
    """
    For each cube, pick the move that minimises
    max(Slice_Twist_Prun, Slice_Flip_Prun) on the resulting state.
    Ties broken by lowest move index.
    """
    twist_mv, flip_mv, frtobr_mv, _, _ = tables
    prun_twist, prun_flip = prun_tables

    # (N, 18) successor coordinates
    new_twist = twist_mv[state["twist"]]
    new_flip = flip_mv[state["flip"]]
    new_frtobr = frtobr_mv[state["frtobr"]]
    new_slice = new_frtobr // 24  # shape (N, 18), in [0, N_SLICE1)

    idx_twist = N_SLICE1 * new_twist + new_slice
    idx_flip = N_SLICE1 * new_flip + new_slice
    cost_twist = prun_twist[idx_twist]
    cost_flip = prun_flip[idx_flip]
    cost = np.maximum(cost_twist, cost_flip)  # (N, 18) uint8

    # argmin with lowest-index tiebreak is default np.argmin behaviour
    labels = cost.argmin(axis=1).astype(np.int32)
    return labels, cost


# ---------------------------------------------------------------------------
# 161-dim features (vectorised)
# ---------------------------------------------------------------------------

def _bin_onehot(values, total, n_bins):
    """
    Vectorised binned one-hot: shape (N, n_bins).

    Matches `_bin_encode` in neural_heuristic.py:
      bin_idx = min(n_bins - 1, value * n_bins // total)
    """
    n = values.shape[0]
    bin_idx = np.minimum((values.astype(np.int64) * n_bins) // total,
                         n_bins - 1).astype(np.int64)
    out = np.zeros((n, n_bins), dtype=np.float32)
    out[np.arange(n), bin_idx] = 1.0
    return out


def extract_features_batch(state):
    """Return (N, 161) matrix matching `extract_coordinate_features`."""
    twist = state["twist"]
    flip = state["flip"]
    frtobr = state["frtobr"]
    urfdlf = state["urfdlf"]
    parity = state["parity"]
    slice_coord = frtobr // 24

    # Binned one-hot
    bt = _bin_onehot(twist, N_TWIST, N_TWIST_BINS)
    bf = _bin_onehot(flip, N_FLIP, N_FLIP_BINS)
    bs = _bin_onehot(slice_coord, CoordCube.N_SLICE1, N_SLICE_BINS)  # N_SLICE1 == total slice states in Phase 1
    bu = _bin_onehot(urfdlf, N_URFDLF, N_URFDLF_BINS)
    bfr = _bin_onehot(frtobr, N_FRTOBR, N_FRTOBR_BINS)

    # Parity one-hot
    parity_oh = np.zeros((parity.shape[0], 2), dtype=np.float32)
    parity_oh[np.arange(parity.shape[0]), parity] = 1.0

    # Scalars (9)
    t_norm = (twist / N_TWIST).astype(np.float32)
    f_norm = (flip / N_FLIP).astype(np.float32)
    s_norm = (slice_coord / CoordCube.N_SLICE1).astype(np.float32)  # note: extract_coordinate_features divides by N_SLICE (495), same value
    p_norm = (parity / N_PARITY).astype(np.float32)
    u_norm = (urfdlf / N_URFDLF).astype(np.float32)
    fr_norm = (frtobr / N_FRTOBR).astype(np.float32)
    twist_mod = ((twist % 729) / 729).astype(np.float32)
    flip_mod = ((flip % 256) / 256).astype(np.float32)
    slice_mod = ((slice_coord % 99) / 99).astype(np.float32)

    scalars = np.stack([t_norm, f_norm, s_norm, p_norm, u_norm, fr_norm,
                        twist_mod, flip_mod, slice_mod], axis=1)

    # Interactions (6)
    parity_f = parity.astype(np.float32)
    interactions = np.stack([
        t_norm * f_norm,
        t_norm * s_norm,
        f_norm * s_norm,
        t_norm * parity_f,
        f_norm * parity_f,
        s_norm * parity_f,
    ], axis=1)

    return np.concatenate(
        [bt, bf, bs, bu, bfr, parity_oh, scalars, interactions], axis=1
    )


# ---------------------------------------------------------------------------
# Model (NumPy MLP + Adam + cosine LR)
# ---------------------------------------------------------------------------

def init_model(input_dim, hidden_sizes, out_dim, rng):
    sizes = [input_dim, *hidden_sizes, out_dim]
    weights, biases = [], []
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        scale = np.sqrt(2.0 / fan_in)  # He init
        w = rng.standard_normal((sizes[i], sizes[i + 1])).astype(np.float32) * scale
        b = np.zeros(sizes[i + 1], dtype=np.float32)
        weights.append(w)
        biases.append(b)
    return weights, biases


def forward_with_cache(X, weights, biases):
    """Forward pass, returning logits and activations for backprop."""
    acts = [X]
    for w, b in zip(weights[:-1], biases[:-1]):
        z = acts[-1] @ w + b
        acts.append(np.maximum(z, 0, dtype=np.float32))
    logits = acts[-1] @ weights[-1] + biases[-1]
    return logits, acts


def softmax_cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    nll = -np.log(probs[np.arange(len(y)), y] + 1e-12)
    return float(nll.mean()), probs


def backward(acts, probs, y, weights):
    n = len(y)
    grad_w = [None] * len(weights)
    grad_b = [None] * len(weights)

    grad_logits = probs.copy()
    grad_logits[np.arange(n), y] -= 1.0
    grad_logits /= n

    grad_w[-1] = acts[-1].T @ grad_logits
    grad_b[-1] = grad_logits.sum(axis=0)
    grad_a = grad_logits @ weights[-1].T

    for i in range(len(weights) - 2, -1, -1):
        grad_z = grad_a * (acts[i + 1] > 0)
        grad_w[i] = acts[i].T @ grad_z
        grad_b[i] = grad_z.sum(axis=0)
        if i > 0:
            grad_a = grad_z @ weights[i].T
    return grad_w, grad_b


class Adam:
    def __init__(self, shapes, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros(s, dtype=np.float32) for s in shapes]
        self.v = [np.zeros(s, dtype=np.float32) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1_c = 1 - self.beta1 ** self.t
        b2_c = 1 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / b1_c
            v_hat = self.v[i] / b2_c
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def cosine_lr(epoch, total, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total))


def top1(logits, y):
    return float((logits.argmax(axis=1) == y).mean())


def top3(logits, y):
    idx = np.argpartition(-logits, 3, axis=1)[:, :3]
    return float(np.any(idx == y[:, None], axis=1).mean())


def per_depth_top1(logits, y, depth):
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
    pred = logits.argmax(axis=1)
    out = []
    for lo, hi in buckets:
        m = (depth >= lo) & (depth <= hi)
        if m.any():
            out.append((lo, hi, int(m.sum()), float((pred[m] == y[m]).mean())))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr-max", type=float, default=4e-3)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260413)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if args.output is None:
        tag = f"{args.samples // 1_000_000}M" if args.samples >= 1_000_000 else f"{args.samples // 1000}k"
        args.output = f"kociemba/kociemba/neural_model_prun{tag}.pkl"
    out_path = Path(args.output)
    meta_path = out_path.with_suffix(".meta.json")

    print("=" * 72)
    print("Prun-greedy training")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k:12s} = {v}")
    print(f"  feature_dim  = {FEATURE_DIM}")
    print(f"  git_commit   = {git_commit()}")
    print()

    wall_start = time.time()

    # 1. Load tables
    print("[1/5] Loading tables")
    tables = load_move_tables()
    prun_tables = load_pruning_tables()

    # 2. Generate dataset
    print(f"[2/5] Generating {args.samples:,} cubes (seed {args.seed})")
    t0 = time.time()
    state = generate_dataset(args.samples, args.seed, tables)
    print(f"  done in {time.time()-t0:.1f}s")

    # 3. Labels
    print("[3/5] Computing prun-greedy labels")
    t0 = time.time()
    labels, cost_mat = prun_greedy_labels(state, tables, prun_tables)
    print(f"  done in {time.time()-t0:.1f}s")
    label_hist = np.bincount(labels, minlength=N_MOVES).tolist()
    print(f"  label distribution: {label_hist}")
    print(f"  mean min-cost: {float(cost_mat.min(axis=1).mean()):.3f}")

    # 4. Features
    print("[4/5] Extracting 161-dim features")
    t0 = time.time()
    X = extract_features_batch(state).astype(np.float32)
    y = labels
    depth = state["depth"]
    print(f"  X shape: {X.shape}, y shape: {y.shape}, "
          f"size: {X.nbytes/1e9:.2f} GB ({time.time()-t0:.1f}s)")

    # 5. Shuffle + split
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(args.samples)
    X = X[perm]
    y = y[perm]
    depth = depth[perm]
    n_val = int(args.samples * args.val_frac)
    n_train = args.samples - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    depth_val = depth[n_train:]
    print(f"  train={n_train:,}  val={n_val:,}")

    # 6. Init model + optimiser
    print(f"[5/5] Training {args.hidden} for {args.epochs} epochs")
    weights, biases = init_model(FEATURE_DIM, args.hidden, N_MOVES,
                                 np.random.default_rng(args.seed + 2))
    opt_w = Adam([w.shape for w in weights], lr=args.lr_max)
    opt_b = Adam([b.shape for b in biases], lr=args.lr_max)

    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    history = []
    best_val_top1 = -1.0
    best_epoch = -1
    best_weights = None
    best_biases = None
    patience_left = args.patience

    for epoch in range(args.epochs):
        lr = cosine_lr(epoch, args.epochs, args.lr_max, args.lr_min)
        opt_w.lr = lr
        opt_b.lr = lr

        shuffle = rng.permutation(n_train)
        X_sh = X_train[shuffle]
        y_sh = y_train[shuffle]

        train_loss = 0.0
        ep_start = time.time()
        for bi in range(n_batches):
            s = bi * args.batch_size
            e = min(s + args.batch_size, n_train)
            xb = X_sh[s:e]
            yb = y_sh[s:e]
            logits, acts = forward_with_cache(xb, weights, biases)
            loss, probs = softmax_cross_entropy(logits, yb)
            train_loss += loss
            grad_w, grad_b = backward(acts, probs, yb, weights)
            opt_w.step(weights, grad_w)
            opt_b.step(biases, grad_b)
        train_loss /= n_batches

        val_logits, _ = forward_with_cache(X_val, weights, biases)
        val_loss, _ = softmax_cross_entropy(val_logits, y_val)
        v_top1 = top1(val_logits, y_val)
        v_top3 = top3(val_logits, y_val)

        took = time.time() - ep_start
        print(f"  epoch {epoch+1:2d}/{args.epochs} "
              f"lr={lr:.4f}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  top1={v_top1*100:6.2f}%  "
              f"top3={v_top3*100:6.2f}%  ({took:.1f}s)",
              flush=True)
        history.append({
            "epoch": epoch + 1, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_top1": v_top1, "val_top3": v_top3,
            "seconds": took,
        })

        if v_top1 > best_val_top1:
            best_val_top1 = v_top1
            best_epoch = epoch + 1
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop (no val-top1 improvement for "
                      f"{args.patience} epochs)")
                break

    print()
    print(f"Best val top-1: {best_val_top1*100:.2f}% at epoch {best_epoch}")

    # Final breakdowns on the best weights
    weights = best_weights
    biases = best_biases
    val_logits, _ = forward_with_cache(X_val, weights, biases)
    final_top1 = top1(val_logits, y_val)
    final_top3 = top3(val_logits, y_val)
    per_depth = per_depth_top1(val_logits, y_val, depth_val)

    print(f"Final top-1 (best checkpoint): {final_top1*100:.2f}%")
    print(f"Final top-3 (best checkpoint): {final_top3*100:.2f}%")
    print("Per-depth top-1 on val:")
    for lo, hi, n, acc in per_depth:
        print(f"  depth [{lo:2d},{hi:2d}]  n={n:5d}  top1={acc*100:6.2f}%")

    # 7. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_payload = {
        "weights": weights,
        "biases": biases,
        "hidden_sizes": args.hidden,
        "feature_mode": "coordinate",
    }
    with open(out_path, "wb") as f:
        pickle.dump(pickle_payload, f)

    wall_end = time.time()

    meta = {
        "output_pickle": str(out_path),
        "sha256": sha256_file(out_path),
        "label_scheme": "prun_greedy (max Slice_Twist_Prun/Slice_Flip_Prun)",
        "feature_mode": "coordinate",
        "feature_dim": FEATURE_DIM,
        "hidden_sizes": args.hidden,
        "n_samples": args.samples,
        "n_train": n_train,
        "n_val": n_val,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "final_val_top1": final_top1,
        "final_val_top3": final_top3,
        "per_depth_val_top1": [
            {"depth_lo": lo, "depth_hi": hi, "n": n, "top1": acc}
            for lo, hi, n, acc in per_depth
        ],
        "label_distribution": label_hist,
        "lr_max": args.lr_max,
        "lr_min": args.lr_min,
        "batch_size": args.batch_size,
        "history": history,
        "wall_seconds": wall_end - wall_start,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved pickle:   {out_path}  ({os.path.getsize(out_path):,} bytes)")
    print(f"Saved metadata: {meta_path}")
    print(f"SHA256:         {meta['sha256']}")
    print(f"Wall time:      {wall_end - wall_start:.1f}s")


if __name__ == "__main__":
    main()
