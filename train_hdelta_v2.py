"""
Option 1 (real attempt): multi-output h-delta predictor with embedding-table
features instead of 32-bin one-hots.

Why this differs from train_hdelta.py:
    The v1 model converged to val_loss == class-prior entropy because the
    161-dim binned features aliased ~68 distinct coordinate values into each
    bin. Within a bin, Delta_h(m) varies, so no function of the binned input
    can predict per-move structure. v2 replaces the input with an embedding
    sum over the three Phase 1 coordinates (twist in [0, 2187), flip in
    [0, 2048), slice in [0, 495)). Each coordinate value gets its own
    learnable vector, so the model can in principle distinguish every
    Phase 1 state it trains on.

Target, loss, mask, and metric are identical to v1:
    - per-move Delta_h in {-1, 0, +1} -> class in {0, 1, 2}
    - masked 3-way softmax-CE, normalized by total legal (state, move) count
    - p_model = fraction of states where argmax over legal P(class=0)
      coincides with a truly progressing move (y == 0)
    - reference rates: p_lex (first lex-legal child progressing) and
      p_top (oracle: min-h child progressing)
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "kociemba"))

from kociemba.pykociemba.cubiecube import CubieCube
from kociemba.pykociemba.coordcube import CoordCube, getPruning

N_SLICE1 = CoordCube.N_SLICE1
N_TWIST = CoordCube.N_TWIST
N_FLIP = CoordCube.N_FLIP
N_MOVES = 18
N_CLASSES = 3


def h_phase1(twist, flip, slice_):
    return max(
        getPruning(CoordCube.Slice_Flip_Prun, N_SLICE1 * flip + slice_),
        getPruning(CoordCube.Slice_Twist_Prun, N_SLICE1 * twist + slice_),
    )


def make_random_coord_cube(rng):
    while True:
        cc = CubieCube()
        cc.setFlip(rng.randint(0, CoordCube.N_FLIP - 1))
        cc.setTwist(rng.randint(0, CoordCube.N_TWIST - 1))
        cc.setURFtoDLB(rng.randint(0, CoordCube.N_URFtoDLB - 1))
        cc.setURtoBR(rng.randint(0, CoordCube.N_URtoBR - 1))
        if (cc.edgeParity() ^ cc.cornerParity()) == 0:
            return cc


def sample_walk_states(rng, n_cubes, walk_len):
    """
    For each cube, take a uniform random walk through Phase 1 and collect
    (twist, flip, slice) indices plus per-move Delta_h labels at every
    visited state.
    """
    max_rows = n_cubes * walk_len
    T = np.zeros(max_rows, dtype=np.int32)
    F = np.zeros(max_rows, dtype=np.int32)
    S = np.zeros(max_rows, dtype=np.int32)
    Y = np.full((max_rows, N_MOVES), -1, dtype=np.int8)
    M = np.zeros((max_rows, N_MOVES), dtype=np.float32)
    rows = 0

    for cube_idx in range(n_cubes):
        cc = make_random_coord_cube(rng)
        coord = CoordCube(cc)
        twist, flip, FRtoBR = coord.twist, coord.flip, coord.FRtoBR
        parent_ax = None

        for _ in range(walk_len):
            slice_ = FRtoBR // 24
            h = h_phase1(twist, flip, slice_)
            if h == 0:
                break

            T[rows] = twist
            F[rows] = flip
            S[rows] = slice_

            legal = []
            for ax in range(6):
                forbidden = parent_ax is not None and (
                    ax == parent_ax or ax == parent_ax - 3
                )
                if forbidden:
                    continue
                for po in range(1, 4):
                    m = 3 * ax + (po - 1)
                    new_twist = CoordCube.twistMove[twist][m]
                    new_flip = CoordCube.flipMove[flip][m]
                    new_FRtoBR = CoordCube.FRtoBR_Move[FRtoBR][m]
                    new_slice = new_FRtoBR // 24
                    hc = h_phase1(new_twist, new_flip, new_slice)
                    Y[rows, m] = (hc - h) + 1
                    M[rows, m] = 1.0
                    legal.append((m, ax, new_twist, new_flip, new_FRtoBR))

            rows += 1
            if not legal:
                break
            m, ax, twist, flip, FRtoBR = legal[rng.randint(0, len(legal) - 1)]
            parent_ax = ax

        if (cube_idx + 1) % 1000 == 0:
            print(f"  sampled {cube_idx + 1}/{n_cubes} cubes, "
                  f"{rows} states", flush=True)

    return T[:rows], F[:rows], S[:rows], Y[:rows], M[:rows]


class EmbedMLP:
    """
    Concatenated-embedding MLP.

    The v1-sum variant failed because E_twist[t] + E_flip[f] + E_slice[s]
    cannot express joint (t, f, s) interactions, and Delta_h(move) is
    fundamentally a joint function of all three coordinates through the
    move tables. Concatenation gives the first hidden layer independent
    slots per coordinate, so it can learn cross-coordinate features.

    Forward pass:
        x = concat(E_t[t], E_f[f], E_s[s])      (B, 3*D)
        h1 = ReLU(x @ W1 + b1)                  (B, H1)
        h2 = ReLU(h1 @ W2 + b2)                 (B, H2)
        logits = h2 @ W3 + b3                   (B, 54)
    """

    def __init__(self, embed_dim=128, hidden1=512, hidden2=256, seed=42):
        rng = np.random.default_rng(seed)
        scale_e = 1.0 / np.sqrt(embed_dim)
        self.E_t = (rng.standard_normal((N_TWIST, embed_dim)).astype(np.float32)
                    * scale_e)
        self.E_f = (rng.standard_normal((N_FLIP, embed_dim)).astype(np.float32)
                    * scale_e)
        self.E_s = (rng.standard_normal((N_SLICE1, embed_dim)).astype(np.float32)
                    * scale_e)

        in_dim = 3 * embed_dim
        self.W1 = (rng.standard_normal((in_dim, hidden1)).astype(np.float32)
                   * np.sqrt(2.0 / in_dim))
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden1, hidden2)).astype(np.float32)
                   * np.sqrt(2.0 / hidden1))
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = (rng.standard_normal((hidden2, N_MOVES * N_CLASSES))
                   .astype(np.float32) * np.sqrt(2.0 / hidden2))
        self.b3 = np.zeros(N_MOVES * N_CLASSES, dtype=np.float32)

        self.embed_dim = embed_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

    def forward(self, t, f, s):
        x = np.concatenate([self.E_t[t], self.E_f[f], self.E_s[s]], axis=1)
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        logits = h2 @ self.W3 + self.b3
        cache = (t, f, s, x, z1, h1, z2, h2)
        return logits, cache

    def loss_and_grads(self, t, f, s, y, mask):
        B = t.shape[0]
        logits, cache = self.forward(t, f, s)
        logits = logits.reshape(B, N_MOVES, N_CLASSES)

        shift = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shift)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        y_safe = np.where(y < 0, 0, y).astype(np.int64)
        true_logp = np.log(np.maximum(
            probs[np.arange(B)[:, None], np.arange(N_MOVES)[None, :], y_safe],
            1e-12,
        ))
        total_mask = float(mask.sum()) + 1e-6
        loss = float(-(true_logp * mask).sum() / total_mask)

        grad_logits = probs.copy()
        grad_logits[np.arange(B)[:, None], np.arange(N_MOVES)[None, :],
                    y_safe] -= 1.0
        grad_logits *= mask[:, :, None]
        grad_logits /= total_mask
        grad_logits = grad_logits.reshape(B, N_MOVES * N_CLASSES)

        (t_, f_, s_, x, z1, h1, z2, h2) = cache
        gW3 = h2.T @ grad_logits
        gb3 = grad_logits.sum(axis=0)
        dh2 = grad_logits @ self.W3.T
        dz2 = dh2 * (z2 > 0).astype(np.float32)
        gW2 = h1.T @ dz2
        gb2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        gW1 = x.T @ dz1
        gb1 = dz1.sum(axis=0)
        dx = dz1 @ self.W1.T  # (B, 3*D)

        D = self.embed_dim
        gE_t_grad = dx[:, :D]
        gE_f_grad = dx[:, D:2 * D]
        gE_s_grad = dx[:, 2 * D:]

        gE_t = np.zeros_like(self.E_t)
        gE_f = np.zeros_like(self.E_f)
        gE_s = np.zeros_like(self.E_s)
        np.add.at(gE_t, t_, gE_t_grad)
        np.add.at(gE_f, f_, gE_f_grad)
        np.add.at(gE_s, s_, gE_s_grad)

        prog = probs[:, :, 0]
        prog = np.where(mask > 0, prog, -np.inf)
        pred_top = prog.argmax(axis=1)
        true_is_prog = (y == 0).astype(np.float32) * mask
        top_hit = true_is_prog[np.arange(B), pred_top]
        p_model = float(top_hit.mean())

        grads = {
            "E_t": gE_t, "E_f": gE_f, "E_s": gE_s,
            "W1": gW1, "b1": gb1,
            "W2": gW2, "b2": gb2,
            "W3": gW3, "b3": gb3,
        }
        return loss, grads, p_model

    def step(self, grads, lr):
        self.E_t -= lr * grads["E_t"]
        self.E_f -= lr * grads["E_f"]
        self.E_s -= lr * grads["E_s"]
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        self.W3 -= lr * grads["W3"]
        self.b3 -= lr * grads["b3"]


def train(model, data, val, epochs, batch, lr):
    T, F, S, Y, M = data
    Tv, Fv, Sv, Yv, Mv = val
    n = T.shape[0]
    history = []

    for epoch in range(epochs):
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs)) + lr * 0.03
        perm = np.random.permutation(n)
        losses, hits = [], []
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss, grads, p_m = model.loss_and_grads(
                T[idx], F[idx], S[idx], Y[idx], M[idx]
            )
            model.step(grads, lr_t)
            losses.append(loss)
            hits.append(p_m)
        train_loss = float(np.mean(losses))
        train_p = float(np.mean(hits))

        v_losses, v_hits = [], []
        for i in range(0, Tv.shape[0], batch):
            loss, _, p_m = model.loss_and_grads(
                Tv[i:i + batch], Fv[i:i + batch], Sv[i:i + batch],
                Yv[i:i + batch], Mv[i:i + batch],
            )
            v_losses.append(loss)
            v_hits.append(p_m)
        val_loss = float(np.mean(v_losses))
        val_p = float(np.mean(v_hits))

        dt = time.time() - t0
        print(f"  epoch {epoch + 1:>2}/{epochs}  "
              f"lr={lr_t:.4f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_p={train_p:.3f}  val_p={val_p:.3f}  "
              f"({dt:.1f}s)", flush=True)
        history.append({
            "epoch": epoch + 1, "lr": lr_t,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_p_model": train_p, "val_p_model": val_p,
        })

    return history


def reference_rates(Y, M):
    """p_lex (first legal move) and p_top (oracle: any progressing move)."""
    n = Y.shape[0]
    lex_hits = 0
    top_hits = 0
    for i in range(n):
        legal = np.where(M[i] > 0)[0]
        if len(legal) == 0:
            continue
        if Y[i, legal[0]] == 0:
            lex_hits += 1
        if np.any(Y[i, legal] == 0):
            top_hits += 1
    return lex_hits / n, top_hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cubes", type=int, default=20000)
    ap.add_argument("--walk-len", type=int, default=30)
    ap.add_argument("--val-cubes", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden1", type=int, default=512)
    ap.add_argument("--hidden2", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output",
                    default="kociemba/kociemba/neural_hdelta_v2.pkl")
    args = ap.parse_args()

    print(f"Sampling training data: {args.n_cubes} cubes "
          f"x walk_len={args.walk_len}")
    rng = random.Random(args.seed)
    t0 = time.time()
    T, F, S, Y, M = sample_walk_states(rng, args.n_cubes, args.walk_len)
    print(f"  train states: {T.shape[0]}  ({time.time() - t0:.1f}s)")

    rng_v = random.Random(args.seed + 1)
    Tv, Fv, Sv, Yv, Mv = sample_walk_states(rng_v, args.val_cubes,
                                            args.walk_len)
    print(f"  val states:   {Tv.shape[0]}")

    p_lex, p_top = reference_rates(Yv, Mv)
    print(f"  val p_lex (lex baseline)   = {p_lex:.4f}")
    print(f"  val p_top (oracle ceiling) = {p_top:.4f}")

    print(f"\nTraining EmbedMLP  embed={args.embed_dim}  "
          f"hidden={args.hidden1},{args.hidden2}")
    model = EmbedMLP(embed_dim=args.embed_dim,
                     hidden1=args.hidden1, hidden2=args.hidden2,
                     seed=args.seed)
    history = train(model, (T, F, S, Y, M), (Tv, Fv, Sv, Yv, Mv),
                    epochs=args.epochs, batch=args.batch, lr=args.lr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "E_t": model.E_t, "E_f": model.E_f, "E_s": model.E_s,
        "W1": model.W1, "b1": model.b1,
        "W2": model.W2, "b2": model.b2,
        "W3": model.W3, "b3": model.b3,
        "embed_dim": args.embed_dim,
        "hidden1": args.hidden1, "hidden2": args.hidden2,
        "n_moves": N_MOVES, "n_classes": N_CLASSES,
        "history": history,
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved model to {out_path}")

    meta = {
        "config": vars(args),
        "train_states": int(T.shape[0]),
        "val_states": int(Tv.shape[0]),
        "val_p_lex": p_lex,
        "val_p_top": p_top,
        "final_val_p_model": history[-1]["val_p_model"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
    }
    meta_path = Path("results") / "hdelta_v2_train_meta.json"
    meta_path.parent.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSummary written to {meta_path}")
    print(f"  val_p_lex   = {p_lex:.4f}")
    print(f"  val_p_top   = {p_top:.4f}")
    print(f"  val_p_model = {history[-1]['val_p_model']:.4f}")


if __name__ == "__main__":
    main()
