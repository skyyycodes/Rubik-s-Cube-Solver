"""
Train a multi-output h-delta predictor for Phase 1 move ordering.

Target: for each of the 18 moves m, predict h(child_m) - h(state) in {-1, 0, +1}.
Loss: masked 3-way cross-entropy per move (mask out moves forbidden by the
consecutive-axis filter given parent_ax).

Sampling distribution: random walks of length `walk_len` through Phase 1 state
space (same distribution used by measure_headroom.py), matching the states that
a real Phase 1 IDA* search visits much more closely than uniformly random
scrambles.

Input features: 161-dim binned coordinate features (same as existing 2M model).
Architecture:    161 -> 256 -> 128 -> 64 -> 18*3 logits.

At inference, mask illegal moves, take per-move argmax class, compute
predicted_delta[m] = argmax - 1, and rank moves by predicted_delta ascending.
The top move is the model's recommended first-to-try child.
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
from kociemba.neural_heuristic import extract_coordinate_features

N_SLICE1 = CoordCube.N_SLICE1
FEATURE_DIM = 161
N_MOVES = 18
N_CLASSES = 3  # delta in {-1, 0, +1} -> class {0, 1, 2}


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


def sample_state_labels(rng, n_cubes, walk_len, seed):
    """Yield (features, labels, mask) for each state visited along walks."""
    X = np.zeros((n_cubes * walk_len, FEATURE_DIM), dtype=np.float32)
    Y = np.full((n_cubes * walk_len, N_MOVES), -1, dtype=np.int8)
    M = np.zeros((n_cubes * walk_len, N_MOVES), dtype=np.float32)
    rows = 0

    for cube_idx in range(n_cubes):
        cc = make_random_coord_cube(rng)
        coord = CoordCube(cc)
        twist, flip, FRtoBR = coord.twist, coord.flip, coord.FRtoBR
        URFtoDLF = coord.URFtoDLF
        parity = coord.parity
        parent_ax = None

        for _ in range(walk_len):
            slice_ = FRtoBR // 24
            h = h_phase1(twist, flip, slice_)
            if h == 0:
                break

            feat = extract_coordinate_features(
                twist, flip, slice_, parity=parity,
                URFtoDLF=URFtoDLF, FRtoBR=FRtoBR,
            )
            X[rows] = feat

            legal_moves = []
            for ax in range(6):
                forbidden = parent_ax is not None and (ax == parent_ax or ax == parent_ax - 3)
                for po in range(1, 4):
                    m = 3 * ax + (po - 1)
                    if forbidden:
                        continue
                    new_twist = CoordCube.twistMove[twist][m]
                    new_flip = CoordCube.flipMove[flip][m]
                    new_FRtoBR = CoordCube.FRtoBR_Move[FRtoBR][m]
                    new_slice = new_FRtoBR // 24
                    hc = h_phase1(new_twist, new_flip, new_slice)
                    delta = hc - h  # in {-1, 0, +1}
                    cls = delta + 1  # -> {0, 1, 2}
                    Y[rows, m] = cls
                    M[rows, m] = 1.0
                    legal_moves.append((m, ax, new_twist, new_flip, new_FRtoBR))

            rows += 1
            if not legal_moves:
                break

            m, ax, twist, flip, FRtoBR = random.Random(
                rng.random()
            ).choice(legal_moves)
            # update URFtoDLF and parity too (needed for features)
            URFtoDLF = CoordCube.URFtoDLF_Move[URFtoDLF][m]
            parity = CoordCube.parityMove[parity][m]
            parent_ax = ax

        if (cube_idx + 1) % 500 == 0:
            print(f"  sampled {cube_idx + 1}/{n_cubes} cubes, "
                  f"{rows} states", flush=True)

    return X[:rows], Y[:rows], M[:rows]


class MLP:
    """Minimal numpy MLP with He init and softmax-CE loss on 18 masked heads."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=(256, 128, 64),
                 out_dim=N_MOVES * N_CLASSES, seed=42):
        rng = np.random.default_rng(seed)
        sizes = [in_dim] + list(hidden) + [out_dim]
        self.W = []
        self.b = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((a, b)).astype(np.float32)
                          * np.sqrt(2.0 / a))
            self.b.append(np.zeros(b, dtype=np.float32))

    def forward(self, x, cache=False):
        activations = [x]
        pre = []
        a = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            pre.append(z)
            if i < len(self.W) - 1:
                a = np.maximum(0, z)
            else:
                a = z
            activations.append(a)
        if cache:
            return a, activations, pre
        return a

    def loss_and_grads(self, x, y, mask):
        """
        x:     (B, 161)
        y:     (B, 18)  int class labels in {0,1,2}; -1 where masked
        mask:  (B, 18)  float 0/1
        """
        B = x.shape[0]
        logits, acts, pre = self.forward(x, cache=True)
        logits = logits.reshape(B, N_MOVES, N_CLASSES)
        # softmax
        logits_shift = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits_shift)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        # gather log-prob of true class (guard -1 with 0 index, then mask)
        y_safe = np.where(y < 0, 0, y).astype(np.int64)
        true_logp = np.log(np.maximum(
            probs[np.arange(B)[:, None], np.arange(N_MOVES)[None, :], y_safe],
            1e-12,
        ))
        total_mask = float(mask.sum()) + 1e-6
        loss = float(-(true_logp * mask).sum() / total_mask)

        # gradient w.r.t. logits: softmax-CE standard, masked, normalized by
        # total legal-move count in the batch (so each legal (state, move)
        # entry contributes equally).
        grad_logits = probs.copy()
        grad_logits[np.arange(B)[:, None], np.arange(N_MOVES)[None, :], y_safe] -= 1.0
        grad_logits *= mask[:, :, None]
        grad_logits /= total_mask
        grad_logits = grad_logits.reshape(B, N_MOVES * N_CLASSES)

        # backprop through MLP
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        da = grad_logits
        for i in reversed(range(len(self.W))):
            z = pre[i]
            a_prev = acts[i]
            if i == len(self.W) - 1:
                dz = da
            else:
                dz = da * (z > 0).astype(np.float32)
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T

        # accuracy metric: per-state, does argmax of predicted progressing
        # (class 0) score rank the right child first?
        # Predicted top: argmax over (m where mask=1) of P(delta=-1)
        prog_scores = probs[:, :, 0]
        prog_scores = np.where(mask > 0, prog_scores, -np.inf)
        pred_top = prog_scores.argmax(axis=1)
        true_is_prog = (y == 0).astype(np.float32) * mask
        has_prog = true_is_prog.sum(axis=1) > 0
        top_hit = true_is_prog[np.arange(B), pred_top]
        # p_model: fraction of states where pred_top is a progressing move
        p_model = top_hit.mean()
        # also: fraction where any progressing move exists
        p_upper = has_prog.mean()

        return loss, grads_W, grads_b, float(p_model), float(p_upper)


def train(X, Y, M, Xv, Yv, Mv, epochs=20, batch=512, lr=3e-3, out_path=None):
    model = MLP()
    n = X.shape[0]
    history = []

    for epoch in range(epochs):
        # cosine lr
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs)) + 1e-4
        perm = np.random.permutation(n)
        losses = []
        hits = []
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb, mb = X[idx], Y[idx], M[idx]
            loss, gW, gb, p_m, _ = model.loss_and_grads(xb, yb, mb)
            for k in range(len(model.W)):
                model.W[k] -= lr_t * gW[k]
                model.b[k] -= lr_t * gb[k]
            losses.append(loss)
            hits.append(p_m)
        train_loss = float(np.mean(losses))
        train_p = float(np.mean(hits))

        # validation
        v_losses = []
        v_hits = []
        for i in range(0, Xv.shape[0], batch):
            xb = Xv[i:i + batch]
            yb = Yv[i:i + batch]
            mb = Mv[i:i + batch]
            loss, _, _, p_m, _ = model.loss_and_grads(xb, yb, mb)
            v_losses.append(loss)
            v_hits.append(p_m)
        val_loss = float(np.mean(v_losses))
        val_p = float(np.mean(v_hits))

        dt = time.time() - t0
        print(f"  epoch {epoch + 1:>2}/{epochs}  "
              f"lr={lr_t:.4f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_p_model={train_p:.3f}  val_p_model={val_p:.3f}  "
              f"({dt:.1f}s)", flush=True)
        history.append({
            "epoch": epoch + 1, "lr": lr_t,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_p_model": train_p, "val_p_model": val_p,
        })

    if out_path:
        data = {
            "W": [w.astype(np.float32) for w in model.W],
            "b": [b.astype(np.float32) for b in model.b],
            "feature_dim": FEATURE_DIM,
            "n_moves": N_MOVES,
            "n_classes": N_CLASSES,
            "hidden_sizes": [256, 128, 64],
            "history": history,
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved model to {out_path}")

    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cubes", type=int, default=5000)
    ap.add_argument("--walk-len", type=int, default=30)
    ap.add_argument("--val-cubes", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="kociemba/kociemba/neural_hdelta.pkl")
    args = ap.parse_args()

    print(f"Sampling training data: {args.n_cubes} cubes x walk_len={args.walk_len}")
    rng = random.Random(args.seed)
    t0 = time.time()
    X, Y, M = sample_state_labels(rng, args.n_cubes, args.walk_len, args.seed)
    print(f"  train states: {X.shape[0]}  ({time.time() - t0:.1f}s)")

    rng_v = random.Random(args.seed + 1)
    Xv, Yv, Mv = sample_state_labels(rng_v, args.val_cubes, args.walk_len, args.seed + 1)
    print(f"  val states:   {Xv.shape[0]}")

    # quick sanity: what is p_lex and p_top on the training distribution?
    # p_lex: first legal move's class == 0
    def rate(yy, mm, chooser):
        hits = 0
        n = yy.shape[0]
        for i in range(n):
            legal = np.where(mm[i] > 0)[0]
            if len(legal) == 0:
                continue
            pick = chooser(legal, yy[i])
            if yy[i, pick] == 0:
                hits += 1
        return hits / n

    p_lex_v = rate(Yv, Mv, lambda legal, y: legal[0])
    # oracle: among legal moves, pick any progressing if exists else first
    def oracle_pick(legal, y):
        for m in legal:
            if y[m] == 0:
                return m
        return legal[0]
    p_top_v = rate(Yv, Mv, oracle_pick)
    print(f"  val p_lex (baseline)       = {p_lex_v:.4f}")
    print(f"  val p_top (oracle ceiling) = {p_top_v:.4f}")

    print(f"\nTraining MLP ({args.epochs} epochs, batch={args.batch}, lr={args.lr})")
    model, history = train(X, Y, M, Xv, Yv, Mv,
                           epochs=args.epochs,
                           batch=args.batch,
                           lr=args.lr,
                           out_path=args.output)

    # final summary
    out = {
        "config": vars(args),
        "train_states": int(X.shape[0]),
        "val_states": int(Xv.shape[0]),
        "val_p_lex": p_lex_v,
        "val_p_top": p_top_v,
        "final_val_p_model": history[-1]["val_p_model"],
        "history": history,
    }
    meta_path = Path("results") / "hdelta_train_meta.json"
    meta_path.parent.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(out, indent=2))
    print(f"\nSummary written to {meta_path}")
    print(f"  val_p_lex   = {p_lex_v:.4f}")
    print(f"  val_p_top   = {p_top_v:.4f}")
    print(f"  val_p_model = {history[-1]['val_p_model']:.4f}")


if __name__ == "__main__":
    main()
