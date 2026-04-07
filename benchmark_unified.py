#!/usr/bin/env python3
"""
Unified benchmark: runs ALL configurations in a single process so that timing
numbers are directly comparable across tables in the paper.

Generates: results/unified_benchmark.json

Estimated runtime: ~35-40 minutes on Apple M-series.

Configurations:
  1. K=1 baseline (instrumented)         ~100 s
  2. K=3 multi-solution (instrumented)   ~520 s
  3. K=5 multi-solution (instrumented)   ~560 s
  4. K=10 multi-solution (instrumented)  ~2300 s  [optional, --include-k10]
  5. K=5 + LUT neural ordering           ~580 s
  6. K=5 + move_order coord features     ~660 s

All use the SAME 200 scrambles (seed 42) generated ONCE.
"""

import sys
import time
import json
import math
import statistics
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'kociemba'))

from kociemba import _solve_python, _solve_python_instrumented
from kociemba.pykociemba import tools


def generate_scrambles(n: int, seed: int = 42) -> list:
    random.seed(seed)
    return [tools.randomCube() for _ in range(n)]


def count_moves(solution: str) -> int:
    return len(solution.strip().split())


def run_config(scrambles, label, max_phase1=1, timeout=5000,
               time_budget_sec=None, use_neural=False,
               neural_strategy='none', neural_prob=0.5,
               neural_depth_cutoff=5, collect_stats=True):
    """Run a single configuration on the given scrambles."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  K={max_phase1}, neural={neural_strategy}, n={len(scrambles)}")
    print(f"{'='*60}")

    move_counts = []
    times = []
    all_nodes = []
    all_phase1_nodes = []
    all_phase2_nodes = []
    failures = 0

    t0 = time.time()
    for i, cube in enumerate(scrambles):
        start = time.time()
        try:
            if collect_stats:
                solution, stats = _solve_python_instrumented(
                    cube, None, 24, max_phase1, timeout, time_budget_sec,
                    use_neural, neural_strategy, neural_prob, neural_depth_cutoff)
            else:
                solution = _solve_python(
                    cube, None, 24, max_phase1, timeout, time_budget_sec,
                    use_neural, neural_strategy, neural_prob, neural_depth_cutoff)
                stats = None
            elapsed = time.time() - start
            moves = count_moves(solution)
            move_counts.append(moves)
            times.append(elapsed * 1000)
            if stats:
                all_nodes.append(stats['nodes_expanded'])
                all_phase1_nodes.append(stats['phase1_nodes'])
                all_phase2_nodes.append(stats['phase2_nodes'])
        except Exception as e:
            failures += 1
            times.append((time.time() - start) * 1000)

        if (i + 1) % 20 == 0:
            elapsed_total = time.time() - t0
            rate = (i + 1) / elapsed_total
            eta = (len(scrambles) - i - 1) / rate
            print(f"  {label}: {i+1}/{len(scrambles)} "
                  f"({elapsed_total:.0f}s elapsed, ETA {eta:.0f}s)")

    wall_time = time.time() - t0
    n = len(move_counts)
    sorted_times = sorted(times) if times else []
    n_t = len(sorted_times)

    # Compute percentiles
    p95_ms = sorted_times[int(0.95 * (n_t - 1))] if n_t > 1 else (sorted_times[0] if n_t else 0)
    p99_ms = sorted_times[int(0.99 * (n_t - 1))] if n_t > 1 else (sorted_times[0] if n_t else 0)

    # Distribution
    le18 = sum(1 for m in move_counts if m <= 18)
    le20 = sum(1 for m in move_counts if m <= 20)
    le21 = sum(1 for m in move_counts if m <= 21)
    count_19_20 = sum(1 for m in move_counts if m in (19, 20))
    count_21 = sum(1 for m in move_counts if m == 21)
    count_22 = sum(1 for m in move_counts if m >= 22)

    result = {
        'label': label,
        'max_phase1': max_phase1,
        'neural_strategy': neural_strategy,
        'total_scrambles': len(scrambles),
        'successful': n,
        'failures': failures,
        'move_stats': {
            'mean': round(statistics.mean(move_counts), 3) if n else 0,
            'median': statistics.median(move_counts) if n else 0,
            'stdev': round(statistics.stdev(move_counts), 4) if n > 1 else 0,
            'min': min(move_counts) if n else 0,
            'max': max(move_counts) if n else 0,
        },
        'time_stats': {
            'mean_ms': round(statistics.mean(times), 1) if times else 0,
            'median_ms': round(statistics.median(times), 1) if times else 0,
            'p95_ms': round(p95_ms, 1),
            'p99_ms': round(p99_ms, 1),
            'max_ms': round(max(times), 1) if times else 0,
        },
        'distribution': {
            'le18': le18,
            'count_19_20': count_19_20,
            'count_21': count_21,
            'count_22_plus': count_22,
            'le20': le20,
            'le20_pct': round(100 * le20 / n, 1) if n else 0,
            'le21': le21,
        },
        'wall_time_sec': round(wall_time, 1),
    }

    if all_nodes:
        result['node_stats'] = {
            'mean_nodes': round(statistics.mean(all_nodes)),
            'median_nodes': round(statistics.median(all_nodes)),
            'mean_phase1_nodes': round(statistics.mean(all_phase1_nodes)),
            'mean_phase2_nodes': round(statistics.mean(all_phase2_nodes)),
        }

    # Store raw paired data for stats
    result['_move_counts'] = move_counts
    result['_times'] = times

    # Print summary
    print(f"\n  {label} DONE ({wall_time:.1f}s)")
    print(f"    Solved: {n}/{len(scrambles)}")
    if n:
        print(f"    Mean moves: {result['move_stats']['mean']:.2f} "
              f"(stdev {result['move_stats']['stdev']:.2f})")
        print(f"    ≤20: {le20}/{n} ({result['distribution']['le20_pct']}%)")
        print(f"    Mean time: {result['time_stats']['mean_ms']:.1f} ms "
              f"(P95={p95_ms:.1f}, max={max(times):.1f})")
    if all_nodes:
        print(f"    Mean nodes: {result['node_stats']['mean_nodes']:,}")

    return result


def paired_t_test(a, b):
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    if n < 2:
        return 0, 1.0, mean_d, mean_d, mean_d
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t_stat = mean_d / se if se > 0 else 0
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))
    z95 = 1.96
    return t_stat, p_value, mean_d, mean_d - z95*se, mean_d + z95*se


def cohens_d(a, b):
    diffs = [ai - bi for ai, bi in zip(a, b)]
    if len(diffs) < 2:
        return 0
    return statistics.mean(diffs) / statistics.stdev(diffs) if statistics.stdev(diffs) > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified benchmark for paper consistency')
    parser.add_argument('--scrambles', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-k10', action='store_true',
                        help='Include K=10 (adds ~40 min)')
    parser.add_argument('--output', type=str, default='results/unified_benchmark.json')
    args = parser.parse_args()

    print(f"Unified Benchmark")
    print(f"  Scrambles: {args.scrambles}, Seed: {args.seed}")
    print(f"  K=10: {'yes' if args.include_k10 else 'no (use --include-k10)'}")
    print(f"  Output: {args.output}")

    # Generate scrambles ONCE
    print(f"\nGenerating {args.scrambles} scrambles (seed={args.seed})...")
    scrambles = generate_scrambles(args.scrambles, args.seed)
    print(f"  Done. First cube: {scrambles[0][:20]}...")

    results = {}
    total_start = time.time()

    # ── 1. K=1 baseline ──
    results['K1'] = run_config(scrambles, 'Baseline K=1', max_phase1=1)

    # ── 2. K=3 ──
    results['K3'] = run_config(scrambles, 'Multi K=3', max_phase1=3)

    # ── 3. K=5 ──
    results['K5'] = run_config(scrambles, 'Multi K=5', max_phase1=5)

    # ── 4. K=10 (optional) ──
    if args.include_k10:
        results['K10'] = run_config(scrambles, 'Multi K=10', max_phase1=10)

    # ── 5. K=5 + move_order (coord features) ──
    results['K5_move_order'] = run_config(
        scrambles, 'K=5 + move_order (coord)',
        max_phase1=5, use_neural=True, neural_strategy='move_order')

    # ── 6. K=5 + LUT ──
    results['K5_lut'] = run_config(
        scrambles, 'K=5 + LUT',
        max_phase1=5, use_neural=True, neural_strategy='lut')

    total_elapsed = time.time() - total_start

    # ── Statistical tests (K=1 vs K=5) ──
    b_moves = results['K1']['_move_counts']
    # For K=5, we only have successful solves which may be < 200
    # Use only paired entries (both successful)
    k5_moves = results['K5']['_move_counts']
    n_paired = min(len(b_moves), len(k5_moves))
    if n_paired >= 10:
        t_stat, p_val, mean_diff, ci_lo, ci_hi = paired_t_test(
            b_moves[:n_paired], k5_moves[:n_paired])
        d = cohens_d(b_moves[:n_paired], k5_moves[:n_paired])
        results['stats_K1_vs_K5'] = {
            't_stat': round(t_stat, 2),
            'p_value': p_val,
            'mean_diff': round(mean_diff, 3),
            'ci_95': [round(ci_lo, 3), round(ci_hi, 3)],
            'cohens_d': round(d, 2),
            'n_paired': n_paired,
        }
        print(f"\n  K=1 vs K=5 paired test (n={n_paired}):")
        print(f"    Δ = {mean_diff:.3f}, t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}")
        print(f"    95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── Clean internal data and save ──
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'scrambles': args.scrambles,
            'seed': args.seed,
            'include_k10': args.include_k10,
            'total_time_sec': round(total_elapsed),
            'python_version': sys.version,
        }
    }
    for key, res in results.items():
        if isinstance(res, dict) and '_move_counts' in res:
            output[key] = {k: v for k, v in res.items() if not k.startswith('_')}
        else:
            output[key] = res

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  UNIFIED BENCHMARK COMPLETE")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
