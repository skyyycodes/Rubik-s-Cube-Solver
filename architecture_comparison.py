#!/usr/bin/env python3
"""
Architecture comparison & LUT benchmark experiment.

Addresses two reviewer objections:
  1. "No wall-clock improvement from neural ordering"
     → Demonstrates LUT-based move ordering achieves wall-clock speedup.
  2. "Simple MLP, not deep learning"
     → Trains wider / deeper models, shows accuracy saturates while
       inference cost increases, proving the overhead barrier is inherent.

Usage:
    PYTHONPATH=kociemba python3 architecture_comparison.py

Outputs:
    results/architecture_comparison.json
"""

import sys
import os
import json
import time
import random
import statistics
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'kociemba'))

from kociemba.neural_heuristic import (
    NeuralMovePredictor, NeuralMoveTrainer, NeuralMoveLUT,
    FEATURE_MODE_COORDINATE, COORDINATE_FEATURE_SIZE, N_MOVES,
    N_TWIST, N_FLIP, N_SLICE, N_URFDLF, N_FRTOBR,
    extract_coordinate_features,
)


# ─────────────────────────────────────────────────────────────────
# Architecture configurations to compare
# ─────────────────────────────────────────────────────────────────
ARCHITECTURES = {
    'mlp_base':  {'hidden_sizes': [256, 128, 64],            'label': 'MLP-3 (256→128→64)'},
    'mlp_wide':  {'hidden_sizes': [512, 256, 128, 64],       'label': 'MLP-4 Wide (512→256→128→64)'},
    'mlp_deep':  {'hidden_sizes': [256, 256, 256, 128, 64],  'label': 'MLP-5 Deep (256→256→256→128→64)'},
}


def count_params(hidden_sizes, input_size=COORDINATE_FEATURE_SIZE, output_size=N_MOVES):
    """Count total trainable parameters."""
    sizes = [input_size] + hidden_sizes + [output_size]
    total = 0
    for i in range(len(sizes) - 1):
        total += sizes[i] * sizes[i + 1]  # weights
        total += sizes[i + 1]              # biases
    return total


def train_architecture(name, config, X_train, y_train, X_val, y_val, epochs=100):
    """Train one architecture and return results."""
    hidden = config['hidden_sizes']
    label = config['label']
    n_params = count_params(hidden)
    print(f"\n{'='*60}")
    print(f"Training: {label}  ({n_params:,} params)")
    print(f"{'='*60}")

    model = NeuralMovePredictor(feature_mode=FEATURE_MODE_COORDINATE)
    model.hidden_sizes = hidden
    trainer = NeuralMoveTrainer(model=model)
    history = trainer.train(X_train, y_train, epochs=epochs, batch_size=256, learning_rate=0.001)

    # Final accuracy
    pred_train = trainer.predict_batch(X_train[:2000])
    train_acc = np.mean(pred_train == y_train[:2000])
    pred_val = trainer.predict_batch(X_val)
    val_acc = np.mean(pred_val == y_val)

    # Top-3 accuracy
    x = X_val.copy()
    for w, b in zip(model.weights[:-1], model.biases[:-1]):
        x = np.maximum(0, np.dot(x, w) + b)
    logits = np.dot(x, model.weights[-1]) + model.biases[-1]
    top3 = np.argsort(logits, axis=1)[:, -3:]
    top3_acc = np.mean([y_val[i] in top3[i] for i in range(len(y_val))])

    # Inference latency (per single sample)
    single_sample = X_val[:1].copy()
    n_timing = 10000
    t0 = time.perf_counter()
    for _ in range(n_timing):
        x = single_sample
        for w, b in zip(model.weights[:-1], model.biases[:-1]):
            x = np.maximum(0, np.dot(x, w) + b)
        _ = np.dot(x, model.weights[-1]) + model.biases[-1]
    inference_us = (time.perf_counter() - t0) / n_timing * 1e6

    # Full predict_move_order latency (includes feature extraction)
    twist_v, flip_v, slice_v = 1000, 1000, 200
    t0 = time.perf_counter()
    for _ in range(n_timing):
        model.predict_move_order(twist_v, flip_v, slice_v, 0, 5000, 3000)
    full_predict_us = (time.perf_counter() - t0) / n_timing * 1e6

    print(f"  Train acc: {train_acc:.3f}  Val acc: {val_acc:.3f}  Top-3: {top3_acc:.3f}")
    print(f"  Forward pass: {inference_us:.1f} µs    Full predict: {full_predict_us:.1f} µs")

    result = {
        'name': name,
        'label': label,
        'hidden_sizes': hidden,
        'n_params': n_params,
        'train_acc': round(float(train_acc), 4),
        'val_acc': round(float(val_acc), 4),
        'top3_acc': round(float(top3_acc), 4),
        'inference_us': round(inference_us, 1),
        'full_predict_us': round(full_predict_us, 1),
        'final_loss': round(float(history[-1]), 4),
    }
    return model, result


def benchmark_lut(model, X_val, y_val):
    """Build LUT from model and measure accuracy degradation + latency."""
    print(f"\n{'='*60}")
    print("Building LUT from base model...")
    print(f"{'='*60}")

    lut = NeuralMoveLUT()
    lut.build_from_model(model)

    # Measure LUT accuracy vs full model on validation data
    # For each validation sample, compare LUT ordering vs model ordering
    n_check = min(5000, len(X_val))
    top1_agreement = 0
    top3_agreement = 0

    for i in range(n_check):
        # Get model's full prediction
        x = X_val[i:i+1]
        for w, b in zip(model.weights[:-1], model.biases[:-1]):
            x = np.maximum(0, np.dot(x, w) + b)
        logits = np.dot(x, model.weights[-1]) + model.biases[-1]
        model_order = np.argsort(logits.flatten())[::-1]

        # Get LUT prediction (need to reconstruct coordinates from features)
        # The validation features are 161-dim; we need raw coords for the LUT.
        # Instead, let's generate random coordinates and compare.
        pass  # We'll do a different comparison below

    # Better: generate random coordinates, compare model vs LUT predictions
    np.random.seed(123)
    n_compare = 10000
    top1_agree = 0
    top3_agree = 0
    for _ in range(n_compare):
        twist = np.random.randint(N_TWIST)
        flip = np.random.randint(N_FLIP)
        slice_c = np.random.randint(N_SLICE)
        parity = np.random.randint(2)
        urfdlf = np.random.randint(N_URFDLF)
        frtobr = np.random.randint(N_FRTOBR)

        model_order = model.predict_move_order(twist, flip, slice_c, parity, urfdlf, frtobr)
        lut_order = lut.lookup(twist, flip, slice_c, parity).tolist()

        if model_order[0] == lut_order[0]:
            top1_agree += 1
        if model_order[0] in lut_order[:3]:
            top3_agree += 1

    top1_agreement = top1_agree / n_compare
    top3_agreement = top3_agree / n_compare

    # LUT lookup latency
    twist_v, flip_v, slice_v = 1000, 1000, 200
    n_timing = 100000
    t0 = time.perf_counter()
    for _ in range(n_timing):
        lut.lookup_as_axis_power(twist_v, flip_v, slice_v, 0)
    lut_us = (time.perf_counter() - t0) / n_timing * 1e6

    print(f"  LUT vs Model top-1 agreement: {top1_agreement:.3f}")
    print(f"  LUT vs Model top-3 agreement: {top3_agreement:.3f}")
    print(f"  LUT lookup latency: {lut_us:.2f} µs")
    print(f"  Build time: {lut.build_time:.1f}s  Memory: {lut.table.nbytes/1024:.0f} KB")

    return {
        'top1_agreement': round(top1_agreement, 4),
        'top3_agreement': round(top3_agreement, 4),
        'lookup_us': round(lut_us, 2),
        'build_time_s': round(lut.build_time, 1),
        'memory_kb': round(lut.table.nbytes / 1024, 0),
        'n_entries': int(np.prod(lut.table.shape[:-1])),
    }


def run_solver_benchmarks(n_scrambles=50, seed=42):
    """Run solver benchmarks comparing baseline, K5-none, K5-move_order, K5-lut."""
    from kociemba import _solve_python_instrumented

    print(f"\n{'='*60}")
    print(f"Solver benchmark: {n_scrambles} scrambles, seed={seed}")
    print(f"{'='*60}")

    # Generate scrambles (matching benchmark.py methodology)
    from kociemba.pykociemba.cubiecube import CubieCube
    from kociemba.pykociemba.facecube import FaceCube

    random.seed(seed)
    scrambles = []
    for _ in range(n_scrambles):
        cube = CubieCube()
        moves_to_apply = 25
        last_axis = -1
        for __ in range(moves_to_apply):
            while True:
                mv = random.randint(0, 17)
                axis = mv // 3
                if axis != last_axis:
                    break
            last_axis = axis
            power = mv % 3
            from kociemba.pykociemba.cubiecube import moveCube
            for ___ in range(power + 1):
                cube.cornerMultiply(moveCube[axis])
                cube.edgeMultiply(moveCube[axis])
        try:
            fc_str = cube.toFaceCube().to_String()
            scrambles.append(fc_str)
        except Exception:
            continue

    configs = [
        {'label': 'Baseline K=1',      'max_phase1': 1, 'use_neural': False, 'strategy': 'none'},
        {'label': 'K5-none',            'max_phase1': 5, 'use_neural': False, 'strategy': 'none'},
        {'label': 'K5-move_order',      'max_phase1': 5, 'use_neural': True,  'strategy': 'move_order'},
        {'label': 'K5-lut',             'max_phase1': 5, 'use_neural': True,  'strategy': 'lut'},
    ]

    results = {}
    for cfg in configs:
        label = cfg['label']
        print(f"\n  Running: {label}...", end='', flush=True)

        move_counts = []
        times = []
        nodes_list = []
        successes = 0

        for i, cube in enumerate(scrambles):
            t0 = time.time()
            try:
                sol, stats = _solve_python_instrumented(
                    cube, None, 24, cfg['max_phase1'], 5000, None,
                    cfg['use_neural'], cfg['strategy'], 0.5, 5
                )
                elapsed = time.time() - t0
                moves = len(sol.strip().split())
                move_counts.append(moves)
                times.append(elapsed * 1000)
                nodes_list.append(stats['nodes_expanded'])
                successes += 1
            except Exception as e:
                elapsed = time.time() - t0
                times.append(elapsed * 1000)

            if (i + 1) % 10 == 0:
                print(f"\r  Running: {label}... {i+1}/{len(scrambles)}", end='', flush=True)

        print(f"\r  {label}: {successes}/{len(scrambles)} solved", flush=True)

        mean_moves = statistics.mean(move_counts) if move_counts else 0
        mean_time = statistics.mean(times) if times else 0
        mean_nodes = statistics.mean(nodes_list) if nodes_list else 0
        pct_le20 = 100 * sum(1 for m in move_counts if m <= 20) / len(move_counts) if move_counts else 0

        results[label] = {
            'successful': successes,
            'total': len(scrambles),
            'mean_moves': round(mean_moves, 2),
            'pct_le20': round(pct_le20, 1),
            'mean_time_ms': round(mean_time, 1),
            'mean_nodes': int(mean_nodes),
            'p95_time_ms': round(sorted(times)[int(0.95 * (len(times) - 1))], 1) if times else 0,
        }

        print(f"    Moves: {mean_moves:.2f}  ≤20: {pct_le20:.1f}%  "
              f"Nodes: {mean_nodes/1e6:.2f}M  Time: {mean_time:.0f}ms")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Architecture comparison experiment')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Training samples (default: 100000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per architecture (default: 100)')
    parser.add_argument('--solver-scrambles', type=int, default=50,
                        help='Number of scrambles for solver benchmark (default: 50)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/architecture_comparison.json')
    args = parser.parse_args()

    output = {}

    # ── Step 1: Generate training data (once, shared by all architectures) ──
    print(f"\n{'='*60}")
    print("Step 1: Generating training data with coordinate features")
    print(f"{'='*60}")

    dummy_model = NeuralMovePredictor(feature_mode=FEATURE_MODE_COORDINATE)
    trainer = NeuralMoveTrainer(model=dummy_model)
    X, y = trainer.generate_training_data(n_samples=args.samples)

    # Split train/val
    n_val = min(10000, len(X) // 10)
    indices = np.random.RandomState(42).permutation(len(X))
    X_train = X[indices[:-n_val]]
    y_train = y[indices[:-n_val]]
    X_val = X[indices[-n_val:]]
    y_val = y[indices[-n_val:]]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Features: {X_train.shape[1]}")

    # ── Step 2: Train each architecture ──
    print(f"\n{'='*60}")
    print("Step 2: Training architectures")
    print(f"{'='*60}")

    arch_results = {}
    models = {}
    for name, config in ARCHITECTURES.items():
        model, result = train_architecture(name, config, X_train, y_train, X_val, y_val,
                                            epochs=args.epochs)
        arch_results[name] = result
        models[name] = model

    output['architectures'] = arch_results

    # Print comparison table
    print(f"\n{'='*60}")
    print("Architecture Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<30}  {'Params':>8}  {'Val Acc':>7}  {'Top-3':>6}  {'Infer µs':>9}  {'Predict µs':>11}")
    print("-" * 85)
    for name in ARCHITECTURES:
        r = arch_results[name]
        print(f"{r['label']:<30}  {r['n_params']:>8,}  {r['val_acc']:>7.3f}  {r['top3_acc']:>6.3f}"
              f"  {r['inference_us']:>9.1f}  {r['full_predict_us']:>11.1f}")

    # ── Step 3: Build LUT from base model and measure accuracy ──
    base_model = models['mlp_base']
    lut_results = benchmark_lut(base_model, X_val, y_val)
    output['lut'] = lut_results

    # ── Step 4: Install base model for solver benchmark ──
    print(f"\n{'='*60}")
    print("Step 4: Installing base model for solver benchmark")
    print(f"{'='*60}")

    model_path = Path(__file__).parent / 'kociemba' / 'kociemba' / 'neural_model.pkl'
    base_model.save(str(model_path))

    # Clear cached models to force reload
    import kociemba.neural_heuristic as nh
    nh._global_move_model = None
    nh._global_move_lut = None

    # Reload
    from kociemba.pykociemba import search as _s
    _s._neural_move_model = nh.get_move_predictor()
    _s._neural_move_lut = None

    # ── Step 5: Solver benchmarks ──
    solver_results = run_solver_benchmarks(
        n_scrambles=args.solver_scrambles,
        seed=args.seed
    )
    output['solver_benchmarks'] = solver_results

    # ── Step 6: Compute derived metrics ──
    if 'K5-none' in solver_results and 'K5-lut' in solver_results:
        k5_none = solver_results['K5-none']
        k5_lut = solver_results['K5-lut']
        k5_mo = solver_results.get('K5-move_order', {})

        speedup_vs_none = k5_none['mean_time_ms'] / k5_lut['mean_time_ms'] if k5_lut['mean_time_ms'] > 0 else 0
        speedup_vs_mo = k5_mo.get('mean_time_ms', 0) / k5_lut['mean_time_ms'] if k5_lut.get('mean_time_ms', 0) > 0 else 0
        node_reduction = k5_none['mean_nodes'] / k5_lut['mean_nodes'] if k5_lut['mean_nodes'] > 0 else 0

        output['derived'] = {
            'lut_speedup_vs_k5_none': round(speedup_vs_none, 2),
            'lut_speedup_vs_move_order': round(speedup_vs_mo, 2),
            'lut_node_reduction': round(node_reduction, 2),
            'lut_wall_clock_improvement': speedup_vs_none > 1.0,
        }

        print(f"\n{'='*60}")
        print("KEY RESULTS")
        print(f"{'='*60}")
        print(f"  LUT speedup vs K5-none (wall-clock): {speedup_vs_none:.2f}×")
        print(f"  LUT speedup vs K5-move_order:        {speedup_vs_mo:.2f}×")
        print(f"  LUT node reduction:                  {node_reduction:.2f}×")
        print(f"  Wall-clock improvement achieved:      {'YES' if speedup_vs_none > 1.0 else 'NO'}")

    # ── Save results ──
    output['metadata'] = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'samples': args.samples,
        'epochs': args.epochs,
        'solver_scrambles': args.solver_scrambles,
        'seed': args.seed,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
