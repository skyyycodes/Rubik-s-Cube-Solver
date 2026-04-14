#!/usr/bin/env python3
"""10M-sample retrain, 30 epochs, 161-dim coordinate features.

Memory-optimized for 16 GB RAM:
- Pre-allocated numpy arrays (no Python list -> numpy conversion spike)
- In-place index-based shuffling (no full X copy per epoch)
- Peak memory ~10 GB (X=6.44 GB + overhead)
"""
import sys, os, time, pickle, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kociemba'))

import numpy as np
np.random.seed(42)

from kociemba.neural_heuristic import (
    NeuralMovePredictor,
    FEATURE_MODE_COORDINATE,
    COORDINATE_FEATURE_SIZE,
    N_MOVES,
)
from kociemba.pykociemba.coordcube import CoordCube
from kociemba.pykociemba.cubiecube import CubieCube, moveCube

MODEL_PATH = 'kociemba/kociemba/neural_model.pkl'
BACKUP_PATH = 'kociemba/kociemba/neural_model_100k.pkl.bak'
N_SAMPLES = 10_000_000
MAX_DEPTH = 25
EPOCHS = 30
BATCH = 1024
LR = 4e-3  # linear-scaled for batch=1024 (was 1e-3 for batch=256)
HIDDEN = [256, 128, 64]

if os.path.exists(MODEL_PATH) and not os.path.exists(BACKUP_PATH):
    shutil.copy(MODEL_PATH, BACKUP_PATH)
    print(f'Backed up existing model to {BACKUP_PATH}', flush=True)

print('=' * 60, flush=True)
print(f'10M RETRAIN: {N_SAMPLES:,} samples x {EPOCHS} epochs (batch {BATCH})', flush=True)
print(f'Feature dim: {COORDINATE_FEATURE_SIZE}', flush=True)
print(f'X size: {N_SAMPLES * COORDINATE_FEATURE_SIZE * 4 / 1e9:.2f} GB', flush=True)
print('=' * 60, flush=True)

# ------------------------------------------------------------------
# [1/3] Data generation (pre-allocated, in-place fill)
# ------------------------------------------------------------------
model = NeuralMovePredictor(feature_mode=FEATURE_MODE_COORDINATE)
model.hidden_sizes = HIDDEN

print('\n[1/3] Allocating arrays...', flush=True)
X = np.empty((N_SAMPLES, COORDINATE_FEATURE_SIZE), dtype=np.float32)
y = np.empty(N_SAMPLES, dtype=np.int64)

print('[1/3] Generating training data...', flush=True)
t0 = time.time()
samples_per_depth = N_SAMPLES // MAX_DEPTH
idx = 0
last_report = time.time()

for depth in range(1, MAX_DEPTH + 1):
    for _ in range(samples_per_depth):
        cube = CubieCube()
        moves = []
        last_axis = -1
        for _ in range(depth):
            while True:
                mv = np.random.randint(N_MOVES)
                axis = mv // 3
                if axis != last_axis:
                    break
            moves.append(mv)
            last_axis = axis
            power = mv % 3
            for _ in range(power + 1):
                cube.cornerMultiply(moveCube[axis])
                cube.edgeMultiply(moveCube[axis])

        coord = CoordCube(cube)
        last_move = moves[-1]
        axis = last_move // 3
        power = last_move % 3
        inv_power = 2 - power
        good_move = axis * 3 + inv_power

        X[idx] = model.extract_features(
            coord.twist, coord.flip, coord.FRtoBR // 24,
            coord.parity, coord.URFtoDLF, coord.FRtoBR
        )
        y[idx] = good_move
        idx += 1

    now = time.time()
    if now - last_report > 30 or depth == MAX_DEPTH:
        print(f'  depth {depth}/{MAX_DEPTH}  {idx:,}/{N_SAMPLES:,}  '
              f'({(now-t0)/60:.1f} min)', flush=True)
        last_report = now

# Trim (in case samples_per_depth rounding left some unfilled)
n_actual = idx
X = X[:n_actual]
y = y[:n_actual]
t_gen = time.time() - t0
print(f'Data gen done: {n_actual:,} samples in {t_gen/60:.1f} min', flush=True)

# ------------------------------------------------------------------
# Split train/val
# ------------------------------------------------------------------
n_val = 10000
X_train, X_val = X[:-n_val], X[-n_val:]
y_train, y_val = y[:-n_val], y[-n_val:]
n_train = len(X_train)
print(f'Train: {n_train:,} | Val: {n_val:,}', flush=True)

# ------------------------------------------------------------------
# [2/3] Custom training loop: in-place shuffle (index permutation only)
# ------------------------------------------------------------------
print('\n[2/3] Initializing model...', flush=True)
model.initialize_random(input_size=COORDINATE_FEATURE_SIZE)
weights = model.weights
biases = model.biases

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

print(f'[2/3] Training {EPOCHS} epochs (batch {BATCH}, lr {LR})...', flush=True)
n_batches = (n_train + BATCH - 1) // BATCH
t1 = time.time()

for epoch in range(EPOCHS):
    perm = np.random.permutation(n_train)
    epoch_loss = 0.0
    for b in range(n_batches):
        s = b * BATCH
        e = min(s + BATCH, n_train)
        batch_idx = perm[s:e]
        xb = X_train[batch_idx]  # small copy (BATCH rows only)
        yb = y_train[batch_idx]

        # Forward
        a0 = xb
        z1 = a0 @ weights[0] + biases[0]; a1 = relu(z1)
        z2 = a1 @ weights[1] + biases[1]; a2 = relu(z2)
        z3 = a2 @ weights[2] + biases[2]; a3 = relu(z3)
        logits = a3 @ weights[3] + biases[3]
        probs = softmax(logits)

        # Loss
        bs = len(yb)
        log_probs = np.log(probs[np.arange(bs), yb] + 1e-8)
        epoch_loss += -log_probs.mean()

        # Backward
        g_logits = probs
        g_logits[np.arange(bs), yb] -= 1
        g_logits /= bs

        gw3 = a3.T @ g_logits; gb3 = g_logits.sum(axis=0)
        ga3 = g_logits @ weights[3].T
        gz3 = ga3 * (a3 > 0)
        gw2 = a2.T @ gz3;     gb2 = gz3.sum(axis=0)
        ga2 = gz3 @ weights[2].T
        gz2 = ga2 * (a2 > 0)
        gw1 = a1.T @ gz2;     gb1 = gz2.sum(axis=0)
        ga1 = gz2 @ weights[1].T
        gz1 = ga1 * (a1 > 0)
        gw0 = a0.T @ gz1;     gb0 = gz1.sum(axis=0)

        weights[0] -= LR * gw0; biases[0] -= LR * gb0
        weights[1] -= LR * gw1; biases[1] -= LR * gb1
        weights[2] -= LR * gw2; biases[2] -= LR * gb2
        weights[3] -= LR * gw3; biases[3] -= LR * gb3

    epoch_loss /= n_batches
    elapsed = time.time() - t1

    # Mini val accuracy (every epoch — only 10k samples, cheap)
    v = X_val
    for w, bi in zip(weights[:-1], biases[:-1]):
        v = relu(v @ w + bi)
    v_logits = v @ weights[-1] + biases[-1]
    v_pred = v_logits.argmax(axis=1)
    val_acc = (v_pred == y_val).mean()

    eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
    print(f'  epoch {epoch+1:2d}/{EPOCHS}  loss={epoch_loss:.4f}  '
          f'val_acc={val_acc:.4f}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m',
          flush=True)

t_train = time.time() - t1
print(f'Training done in {t_train/60:.1f} min', flush=True)

# ------------------------------------------------------------------
# [3/3] Save
# ------------------------------------------------------------------
print('\n[3/3] Saving model...', flush=True)
model.weights = weights
model.biases = biases
model.save(MODEL_PATH)

total = time.time() - t0
print(f'\n=== DONE ===', flush=True)
print(f'TOTAL: {total/60:.1f} min ({total/3600:.2f} hr)', flush=True)
print(f'  Data gen: {t_gen/60:.1f} min ({100*t_gen/total:.0f}%)', flush=True)
print(f'  Training: {t_train/60:.1f} min ({100*t_train/total:.0f}%)', flush=True)
print(f'  Final val acc: {val_acc:.4f}', flush=True)
