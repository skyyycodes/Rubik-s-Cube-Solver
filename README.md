# Rubik's Cube Solver — Multi-Solution Search & Neural Move Ordering

> **Research paper:** *Multi-Solution Search and the Limits of Neural Guidance in Rubik's Cube Solving*
> Accepted at **CIIR 2026** (Springer LNNS series) — Paper ID 890
> Authors: Akash Chakraborty & Atal Chaudhuri, Sister Nivedita University, Kolkata

This repository contains **everything** needed to reproduce the research paper: the enhanced solver, all training pipelines, benchmarks, frozen result artifacts, figure generation, and three interactive front-ends (camera, 3D GUI, web app).

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Key Results](#key-results)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Interactive Solvers](#interactive-solvers)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Training Neural Models](#training-neural-models)
- [Evaluation & Analysis Scripts](#evaluation--analysis-scripts)
- [Project Structure](#project-structure)
- [The Kociemba Module (Enhanced)](#the-kociemba-module-enhanced)
- [Frozen Artifacts](#frozen-artifacts)
- [Figures](#figures)
- [Research Paper](#research-paper)
- [Dependencies](#dependencies)
- [License](#license)

---

## What This Project Does

This project extends Herbert Kociemba's **Two-Phase Algorithm** for solving Rubik's Cube in two ways:

1. **Multi-Solution Phase 1 Search (K-best):** Instead of accepting the first Phase 1 solution, the solver collects up to *K* Phase 1 endpoints and picks the one that yields the shortest total (Phase 1 + Phase 2) solution. This is a purely algorithmic improvement — no machine learning required.

2. **Neural Move Ordering in IDA\*:** A lightweight neural network predicts which move to try first at each search node, aiming to reduce the number of nodes IDA\* expands. The network is distilled into a **32K-entry lookup table (LUT)** for O(1) per-node inference (~1.5 μs).

The paper's central finding is a **carefully measured negative result**: while multi-solution search delivers clear improvements, neural move ordering provides only a marginal ~1.13× node reduction because Kociemba's pruning tables already supply near-optimal move ordering. The paper diagnoses *why* — it is a feature-resolution ceiling, not an information-theoretic one — and proves this via controlled experiments (h-delta retraining, oracle h_sort, headroom measurement).

---

## Key Results

All numbers from the canonical 200-cube benchmark (seed 42, 5 s budget):

| Configuration | Mean Moves | ≤20-Move Rate | Mean Time | Mean Nodes |
|---|---|---|---|---|
| **Baseline K=1** | 20.77 | 28.0% | 582 ms | 787,601 |
| **K=5 (no neural)** | 20.23 | 57.5% | 5,103 ms | 6,793,021 |
| **K=5 + LUT neural** | 20.27 | 56.3% | 5,234 ms | 6,002,669 |

- **Multi-solution K=5** improves mean moves by **0.53** (paired *t* = 11.16, Cohen's *d*_z = 0.79, 95% CI [0.44, 0.62])
- **Neural LUT** reduces nodes by ~1.13× but does **not** improve wall-clock time
- **Oracle h_sort** achieves ~1.66× node reduction — proving headroom exists but lightweight features can't reach it
- Four-seed validation (seeds 42, 123, 456, 789) confirms Δ = 0.40 ± 0.09

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│                                                                 │
│  main.py          cube3d.py           webapp.py                 │
│  (Camera/Manual)  (3D Matplotlib)     (Streamlit Web)           │
└──────┬──────────────┬──────────────────┬────────────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               kociemba/ (Enhanced Local Package)                │
│                                                                 │
│  __init__.py ── solve() API with multi-solution + neural args   │
│       │                                                         │
│       ├── pykociemba/                                           │
│       │    ├── search.py ─── Two-phase IDA* with:               │
│       │    │                  • K-best Phase 1 collection       │
│       │    │                  • Neural move ordering hook        │
│       │    │                  • Node counting instrumentation    │
│       │    │                  • Time-bounded anytime search      │
│       │    ├── coordcube.py ─ Coordinate representation +       │
│       │    │                  pruning tables (40 MB in RAM)      │
│       │    ├── cubiecube.py ─ Cubie-level representation        │
│       │    └── tools.py ──── Random cube generation             │
│       │                                                         │
│       ├── neural_heuristic.py ── NeuralMovePredictor            │
│       │    • 161-dim binned coordinate feature extraction       │
│       │    • LUT builder (32K entries, 576 KB, O(1) lookup)     │
│       │    • Training pipeline (batch SGD, validation)          │
│       │                                                         │
│       ├── neural_model.pkl ──── Trained model (2M samples)      │
│       ├── neural_hdelta.pkl ─── h-delta v1 predictor            │
│       ├── neural_hdelta_v2.pkl ─ h-delta v2 (embedding-based)   │
│       │                                                         │
│       └── ckociemba/ ── C backend (~60× faster, K=1 only)      │
│            └── h_sort variant (oracle move ordering)            │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
│                                                                 │
│  train_neural_heuristic.py ─ Proxy-label training (50K–100K)    │
│  train_prun_greedy.py ────── Pruning-greedy labels (2M)         │
│  train_hdelta.py ──────────── h-delta v1 (binned features)      │
│  train_hdelta_v2.py ────────── h-delta v2 (embedding features)  │
│  train_10m.py ──────────────── 10M-sample production retrain    │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Evaluation & Figures                             │
│                                                                 │
│  benchmark.py ──────────── Full benchmark suite (all configs)   │
│  run_benchmark.py ──────── Quick 200-cube comparison            │
│  benchmark_unified.py ──── All-in-one (~40 min)                 │
│  eval_neural_acc.py ────── Held-out accuracy evaluation         │
│  measure_headroom.py ───── p_top vs p_lex headroom analysis     │
│  architecture_comparison.py ─ MLP width/depth ablation          │
│  generate_figures.py ────── Publication-quality PDF figures      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/AkashChakraborty03/Rubik-s-Cube-Solver.git
cd Rubik-s-Cube-Solver

# Install dependencies
pip install numpy matplotlib streamlit opencv-python

# Solve a cube from the command line (Python API)
python3 -c "
from kociemba.kociemba import solve
print(solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD',
            max_phase1_solutions=5, time_budget_sec=2.0))
"

# Launch the web app
streamlit run webapp.py

# Launch the 3D interactive solver
python3 cube3d.py

# Launch the camera-based solver (requires webcam)
python3 main.py
```

---

## Installation

### Prerequisites
- **Python 3.10+**
- A working webcam (only for `main.py` camera mode)

### Install Dependencies

```bash
pip install numpy matplotlib streamlit opencv-python
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 -c "from kociemba.kociemba import solve; print(solve('UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'))"
# Should print empty string (already solved)
```

---

## Interactive Solvers

### 1. Camera-Based Solver (`main.py`)

Real-time Rubik's Cube solver using your webcam. Point the cube at the camera, and it detects colors via HSV analysis, then overlays directional arrows showing each move.

```bash
python3 main.py
```

**Controls:**
| Key | Action |
|---|---|
| `m` | Toggle manual / camera mode |
| `3` | Switch to 3D input mode |
| `1`–`6` | Set sticker color (manual mode) |
| `Tab` / `Shift+Tab` | Next / previous face |
| `Enter` | Solve |
| `Space` | Next move in solution |
| `r` | Reset |
| `q` | Quit |

### 2. 3D Interactive Solver (`cube3d.py`)

Matplotlib-powered 3D cube visualization. Click stickers to paint them, then solve.

```bash
python3 cube3d.py
```

**Controls:**
| Key | Action |
|---|---|
| Click | Select sticker |
| `1`–`6` | Paint selected sticker |
| Arrow keys | Navigate within face |
| `Enter` | Solve (choose mode: baseline, K=5, anytime, neural LUT) |
| `g` | Toggle guided input mode |
| `r` | Reset |

### 3. Web App (`webapp.py`)

Browser-based solver built with Streamlit. Color picker grid, multiple solve modes, no camera needed.

```bash
streamlit run webapp.py
```

Open `http://localhost:8501` in your browser.

---

## Reproducing Paper Results

### Quick Reproduction (5 minutes)

```bash
python3 run_benchmark.py
```

Runs 200 cubes (seed 42) with Baseline K=1, K=5, and K=5+LUT. Writes results to `benchmark_results.json`.

### Full Benchmark Suite (~40 minutes)

```bash
# Canonical 200-cube comparison (Table 1 in paper)
python3 benchmark.py --scrambles 200 --compare-baseline --max-phase1 5 --seed 42 -v

# Multi-seed validation (4 seeds × 200 cubes)
for seed in 42 123 456 789; do
  python3 benchmark.py --scrambles 200 --compare-baseline --max-phase1 5 --seed $seed \
    --output-json results/benchmark_seed${seed}.json -v
done

# Neural strategy comparison
python3 benchmark.py --scrambles 200 \
  --neural-strategies none move_order probabilistic adaptive \
  --max-phase1 5 --seed 42 --output-json results/neural_strategies.json -v

# Hard-case benchmark (long scrambles)
python3 benchmark.py --scrambles 50 --hard-cases --max-phase1 5 --seed 42 \
  --output-json results/hard_cases.json -v

# Anytime time-budget sweep
python3 benchmark.py --scrambles 200 --time-budgets 0.1 0.3 0.5 1.0 2.0 5.0 \
  --max-phase1 5 --seed 42 -v
```

### Unified All-in-One Benchmark

```bash
python3 benchmark_unified.py --scrambles 200 --seed 42
# Runs K=1, K=3, K=5, K=10, K=5+neural, K=5+LUT sequentially
# Output: results/unified_benchmark.json
```

### C Backend Comparison

```bash
python3 benchmark_c_backend.py    # C vs Python solver speed
python3 bench_c_hsort.py          # Oracle h_sort move ordering
```

### Generate Publication Figures

```bash
python3 generate_figures.py
# Outputs to figures/ (PDF + PNG, 300 DPI)
```

Generates all 7 figures used in the paper:
- `fig_move_distribution.pdf` — Move-count histogram (K=1 vs K=5)
- `fig_k_scaling.pdf` — K-value scaling analysis
- `fig_neural_comparison.pdf` — Neural strategy comparison
- `fig_node_expansion.pdf` — Node expansion by strategy
- `fig_anytime_curve.pdf` — Quality vs time budget
- `fig_hard_cases.pdf` — Hard-case performance
- `fig_multi_seed.pdf` — Multi-seed consistency

---

## Training Neural Models

All models ship pre-trained in `kociemba/kociemba/*.pkl`. To retrain from scratch:

### Proxy-Label Baseline (diagnostic only)

```bash
# 50K cubes, inverse-move labels, ~5 min
python3 train_neural_heuristic.py --n-cubes 50000 --epochs 30
# Produces: ~5.6% top-1 accuracy (essentially random)
```

### Pruning-Greedy Labels (production model)

```bash
# 2M cubes, pruning-table-greedy labels, ~3 hours on CPU
python3 train_prun_greedy.py
# Produces: kociemba/kociemba/neural_model.pkl
# Expected: 43.94% top-1, 64.48% top-3
```

### h-Delta Predictors (controlled negative results)

```bash
# v1: binned features, 150K cubes — converges to class prior
python3 train_hdelta.py --n-cubes 5000

# v2: embedding features, 3M cubes — still converges to class prior
python3 train_hdelta_v2.py --n-cubes 100000
```

These are documented negative results: both models learn the marginal class distribution (prog/plateau/worse ≈ 20/52/28%) and cannot learn move-conditional structure beyond it.

### 10M Production Retrain (memory-optimized)

```bash
# 10M samples, 30 epochs, ~10 GB peak RAM
python3 train_10m.py
```

---

## Evaluation & Analysis Scripts

### Held-Out Neural Accuracy

```bash
python3 eval_neural_acc.py
# Evaluates all neural_model*.pkl on 10K fresh cubes
# Reports top-1, top-3, per-depth-bucket accuracy
```

### Move-Ordering Headroom Measurement

```bash
python3 measure_headroom.py
# Measures p_top (oracle), p_lex (default), gap on 6,000 states
# Key result: p_top = 0.652, p_lex = 0.196, gap = 0.456
```

### Architecture Comparison (MLP width/depth)

```bash
PYTHONPATH=kociemba python3 architecture_comparison.py
# Trains MLP-3, MLP-4 Wide, MLP-5 Deep
# Shows accuracy saturates while inference cost grows
# Output: results/architecture_comparison.json
```

---

## Project Structure

```
Rubik-s-Cube-Solver/
│
├── main.py                      # Camera/manual input solver (OpenCV)
├── cube3d.py                    # 3D interactive solver (Matplotlib)
├── webapp.py                    # Web app solver (Streamlit)
├── color_detect.py              # HSV color detection for camera
├── maker.py                     # Cube state manipulation & validation
├── MYcubePreview.py             # Side-panel cube state visualization
├── Info.py                      # Solution arrow overlay for camera mode
│
├── kociemba/                    # Enhanced Kociemba solver (local package)
│   └── kociemba/
│       ├── __init__.py          # solve() API: multi-solution + neural + anytime
│       ├── neural_heuristic.py  # Neural predictor, LUT builder, trainer (1066 lines)
│       ├── neural_model.pkl     # Trained model (2M samples, pruning-greedy labels)
│       ├── neural_hdelta.pkl    # h-delta v1 (binned features) — negative result
│       ├── neural_hdelta_v2.pkl # h-delta v2 (embedding features) — negative result
│       ├── neural_model_coord.pkl  # Coordinate-feature variant
│       ├── neural_model_scalar.pkl # Scalar-feature variant (5.6% accuracy)
│       ├── pykociemba/          # Pure Python Kociemba implementation
│       │   ├── search.py        # Two-phase IDA* with neural hooks (747 lines)
│       │   ├── coordcube.py     # Coordinate cube + pruning tables
│       │   ├── cubiecube.py     # Cubie-level cube representation
│       │   ├── facecube.py      # Facelet (sticker) representation
│       │   ├── symmetry.py      # Symmetry group operations
│       │   ├── tools.py         # Random cube generation, utilities
│       │   └── prunetables/     # Precomputed pruning tables
│       ├── ckociemba/           # C backend (~60× faster, K=1 only)
│       └── cprunetables/        # C pruning tables (memory-mapped)
│
├── benchmark.py                 # Full benchmark suite (641 lines)
├── run_benchmark.py             # Quick 200-cube comparison
├── benchmark_unified.py         # All-in-one benchmark (~40 min)
├── benchmark_c_backend.py       # C vs Python speed comparison
├── bench_c_hsort.py             # Oracle h_sort experiment
│
├── train_neural_heuristic.py    # Proxy-label training pipeline
├── train_prun_greedy.py         # Pruning-greedy training (production)
├── train_hdelta.py              # h-delta v1 training
├── train_hdelta_v2.py           # h-delta v2 training (embeddings)
├── train_10m.py                 # 10M-sample retrain
│
├── eval_neural_acc.py           # Held-out accuracy evaluation
├── measure_headroom.py          # p_top vs p_lex headroom analysis
├── architecture_comparison.py   # MLP width/depth ablation
├── generate_figures.py          # Publication figure generation (641 lines)
│
├── results/                     # Frozen JSON benchmark artifacts
│   ├── benchmark_200_retrained.json   # Canonical Table 1 data
│   ├── prun_greedy_2M_meta.json       # Training metadata (43.94% top-1)
│   ├── headroom_measurement.json      # p_top=0.652, p_lex=0.196
│   ├── hdelta_train_meta.json         # h-delta v1 training log
│   ├── hdelta_v2_train_meta.json      # h-delta v2 training log
│   ├── architecture_comparison.json   # MLP ablation results
│   ├── h_sort_benchmark.json          # Oracle h_sort results
│   ├── unified_benchmark.json         # Unified benchmark output
│   ├── hard_cases.json                # Hard-case benchmark
│   ├── benchmark_seed*.json           # Multi-seed validation (×4)
│   └── MODEL_PROVENANCE.md            # Model version tracking
│
├── figures/                     # Publication-quality figures (PDF + PNG)
│   ├── fig_move_distribution.pdf
│   ├── fig_k_scaling.pdf
│   ├── fig_neural_comparison.pdf
│   ├── fig_node_expansion.pdf
│   ├── fig_anytime_curve.pdf
│   ├── fig_hard_cases.pdf
│   └── fig_multi_seed.pdf
│
├── main.tex                     # Research paper source (Springer SVMult LaTeX)
├── main.pdf                     # Compiled paper (33 pages)
├── svmult.cls                   # Springer document class
├── svind.ist                    # Springer index style
│
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js config (pdflatex compilation)
└── RubiksCubeSolver.spec        # PyInstaller spec (standalone build)
```

---

## The Kociemba Module (Enhanced)

The `kociemba/` directory contains a **modified fork** of the standard [`kociemba`](https://github.com/muodov/kociemba) Python package. Our modifications:

### What We Changed

1. **Multi-Solution Phase 1** (`search.py`): Instead of returning after the first Phase 1 solution, the search collects up to *K* solutions and selects the one yielding the shortest total path.

2. **Neural Move Ordering Hook** (`search.py`): At each IDA\* node, the solver can optionally reorder child moves based on a neural predictor's ranking before expansion.

3. **Instrumentation** (`search.py`, `__init__.py`): Node counters, timing breakdowns, and per-solve statistics for benchmarking.

4. **Time-Bounded Anytime Mode** (`__init__.py`): The solver keeps searching for better solutions until a wall-clock budget expires.

5. **Neural Heuristic Module** (`neural_heuristic.py`): Complete training, inference, and LUT-distillation pipeline for the neural move predictor.

### Solver API

```python
from kociemba.kociemba import solve

# Basic solve (K=1, like original Kociemba)
solution = solve("DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD")

# Multi-solution K=5
solution = solve(
    "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD",
    max_phase1_solutions=5,
    time_budget_sec=2.0,
)

# With neural LUT ordering
solution = solve(
    "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD",
    max_phase1_solutions=5,
    neural_strategy="move_order",
    time_budget_sec=5.0,
)
```

### Cube String Format

The solver uses the standard Kociemba facelet string: 54 characters in **URFDLB** face order. Each character is one of `U R F D L B` representing the color of that sticker (named by which face's center it matches).

```
         U1 U2 U3
         U4 U5 U6
         U7 U8 U9

L1 L2 L3 F1 F2 F3 R1 R2 R3 B1 B2 B3
L4 L5 L6 F4 F5 F6 R4 R5 R6 B4 B5 B6
L7 L8 L9 F7 F8 F9 R7 R8 R9 B7 B8 B9

         D1 D2 D3
         D4 D5 D6
         D7 D8 D9
```

A solved cube is: `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`

---

## Frozen Artifacts

Every quantitative claim in the paper traces back to a **frozen JSON file** in `results/`. These files are never overwritten by reruns — they are the canonical ground truth.

| Artifact | Key Numbers |
|---|---|
| `benchmark_200_retrained.json` | K=1: mean 20.77, K=5: mean 20.23, LUT: mean 20.27 |
| `prun_greedy_2M_meta.json` | Top-1: 43.94%, Top-3: 64.48% |
| `headroom_measurement.json` | p_top = 0.652, p_lex = 0.196, gap = 0.456 |
| `hdelta_train_meta.json` | v1 converges to 0.932 nats (≈ class prior) |
| `hdelta_v2_train_meta.json` | v2 converges to 0.9252 nats (still class prior) |
| `arch_comp_replay_20260413.json` | MLP-3/4/5: 10.82%/13.41%/13.41% top-1 |

---

## Figures

All figures are generated from frozen JSON artifacts via `generate_figures.py`. They are deterministic — running the script again produces identical output.

| Figure | Description | Source Data |
|---|---|---|
| `fig_move_distribution` | Move-count histogram K=1 vs K=5 | `benchmark_200_retrained.json` |
| `fig_k_scaling` | Mean moves & ≤20-rate vs K | `unified_benchmark.json` |
| `fig_neural_comparison` | Neural strategy comparison | `neural_strategies_retrained.json` |
| `fig_node_expansion` | Node count bar chart | `unified_benchmark.json` |
| `fig_anytime_curve` | Quality vs time budget | `anytime_sweep.json` |
| `fig_hard_cases` | Hard scramble performance | `hard_cases.json` |
| `fig_multi_seed` | 4-seed consistency | `benchmark_seed*.json` |

---

## Research Paper

The paper source is `main.tex`, formatted with the **Springer SVMult** LaTeX template (`svmult.cls`) for the LNNS series.

### Compiling

```bash
pdflatex main.tex && pdflatex main.tex   # Two passes for cross-references
```

Produces `main.pdf` (33 pages). Requires a LaTeX distribution with `mathptmx`, `helvet`, `courier`, `hyperref`, `graphicx`, `amsmath`, `booktabs`.

### Paper Structure

| Section | Content |
|---|---|
| 1. Introduction | Three research questions, novelty claims |
| 2. Related Work | Kociemba, DeepCubeA, neural heuristics (2019–2024) |
| 3. Methods | Two-phase algorithm, K-best search, neural pipeline, formal results |
| 4. Experimental Setup | 200-cube benchmark, feature extraction, training regimes |
| 5. Results & Discussion | K-scaling, neural comparison, h-delta diagnosis, LUT distillation |
| 6. Conclusion | Limitations (practical + theoretical), future work |

---

## Dependencies

| Package | Used By | Purpose |
|---|---|---|
| `numpy` | Everything | Numerical computation |
| `matplotlib` | `cube3d.py`, `generate_figures.py` | Visualization & figure generation |
| `streamlit` | `webapp.py` | Web application framework |
| `opencv-python` | `main.py`, `color_detect.py` | Camera input & color detection |

All neural training uses **pure NumPy** — no PyTorch/TensorFlow required. Models are serialized as Python pickles.

### Optional

- **LaTeX distribution** (for compiling `main.tex`)
- **C compiler** (for building the C backend in `kociemba/ckociemba/` — not required, Python fallback is automatic)

---

## License

The Kociemba solver module is based on [muodov/kociemba](https://github.com/muodov/kociemba) (GPL-2.0). See `kociemba/LICENSE` for details.
