# Neural Move Predictor — Model Provenance

Deployed pickle: [`kociemba/kociemba/neural_model.pkl`](../kociemba/kociemba/neural_model.pkl)

## Current deployed artifact

| Field | Value |
| --- | --- |
| SHA256 | `2c8251c87d66f864a1f66bde4c3448ca004810f89d4b9d83b7608aa9133e7ea0` |
| Size | 668,947 bytes |
| `feature_mode` | `coordinate` |
| Input dim | 161 |
| `hidden_sizes` | `[256, 128, 64]` |
| Output dim | 18 |
| Training script | [`../train_prun_greedy.py`](../train_prun_greedy.py) |
| Training command | `.venv/bin/python train_prun_greedy.py --samples 2000000 --epochs 30 --output /tmp/prun_greedy_2M.pkl` |
| Samples | 2,000,000 |
| Epochs | 30 |
| Optimizer | Adam + cosine LR decay (4e-3 → 1e-4) |
| Labels | Pruning-greedy (argmin over `max(Slice_Twist_Prun, Slice_Flip_Prun)` across 18 moves) |
| Wall time | 10,088.5 s (~2.8 h CPU) |
| Training metadata | [`prun_greedy_2M_meta.json`](prun_greedy_2M_meta.json) |

## Reported accuracy

- **43.94%** top-1 / **64.48%** top-3 on native pruning-greedy labels (epoch 29 validation).
- **15.27%** overall top-1 on proxy-label held-out eval (random baseline 5.56%) — the drop is a label-scheme difference, not a regression. See [`../eval_neural_acc.py`](../eval_neural_acc.py) for the held-out evaluator.
- Per-depth native top-1: `[1,5]=88.99%`, `[6,10]=33.66%`, `[11,15]=29.81%`, `[16,20]=32.80%`, `[21,25]=34.29%`.

## Benchmark contribution

See [`benchmark_200_retrained.json`](benchmark_200_retrained.json). Relative to plain K=5 (no neural), the LUT variant built from this model delivers a **1.13× node reduction** and solves **1 fewer cube** within a 5 s budget. We do not recommend deploying the LUT.

## Origin of the paper's prior 83.3% claim

That number is **not** present in any pickle, training log, or result JSON in this repository. A full replay of the original training recipe (`architecture_comparison.py --samples 100000 --epochs 100 --seed 42`) tops out around 13% validation accuracy — see [`arch_comp_replay_20260413.json`](arch_comp_replay_20260413.json). The paper has been realigned with the honest 43.94% / 1.13× numbers under the "Path Y" edit pass.

## Historical backups (removed)

- `kociemba/kociemba/neural_model.pkl.arch_replay_backup` — prior deployed pickle (scalar features, ~10% top-1). Removed.
- `kociemba/kociemba/neural_model_100k.pkl.bak` — duplicate of the above. Removed.
