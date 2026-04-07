#!/usr/bin/env python3
"""
Benchmark script for improved Kociemba algorithm.

This script measures the performance improvement from:
1. Multi-Solution Phase 1 Search
2. Neural Network Move Ordering strategies (A/B/C)
3. Time-bounded anytime mode

Includes: instrumentation (node counts), hard-case generation,
          statistical validation (paired t-test, bootstrap CI, Cohen's d).

Usage:
    # Original benchmarks (backward compatible):
    python benchmark.py --scrambles 200 --compare-baseline --max-phase1 5 --seed 42 -v
    python benchmark.py --scrambles 200 --time-budgets 0.1 0.3 0.5 1.0 --max-phase1 5 --seed 42

    # Neural strategy comparison:
    python benchmark.py --scrambles 200 --neural-strategies none move_order probabilistic adaptive \\
        --max-phase1 5 --seed 42 --output-json results/neural_strategies.json -v

    # Hard-case benchmark:
    python benchmark.py --scrambles 50 --hard-cases --max-phase1 5 --seed 42 \\
        --output-json results/hard_cases.json -v
"""

import argparse
import time
import math
import statistics
import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add kociemba to path
sys.path.insert(0, str(Path(__file__).parent / 'kociemba'))

from kociemba import solve, _solve_python, _solve_python_instrumented
from kociemba.pykociemba import tools


# ─────────────────────────────────────────────────────────────────
# Scramble / hard-case generation
# ─────────────────────────────────────────────────────────────────

def generate_scrambles(n: int, seed: int = None) -> list:
    """Generate n random cube scrambles."""
    if seed:
        random.seed(seed)
    return [tools.randomCube() for _ in range(n)]


def generate_hard_cases(n: int, seed: int = 42) -> list:
    """
    Generate cubes that are expected to be hard for the Two-Phase solver.

    Strategy: apply exactly 20 random moves (no consecutive same-axis),
    ensuring the cube is deeply scrambled but with a known 20-move upper
    bound.  Also includes the superflip (all edge orientations flipped)
    if possible.
    """
    from kociemba.pykociemba.cubiecube import CubieCube, moveCube

    rng = random.Random(seed)
    cubes = []

    # Include superflip (all edge orientations flipped, everything else solved)
    try:
        superflip = CubieCube()
        superflip.eo = [1] * 12  # flip all edges
        fc_str = superflip.toFaceCube().to_String()
        cubes.append(fc_str)
    except Exception:
        pass

    # Generate deep scrambles (exactly 20 HTM moves)
    while len(cubes) < n:
        cube = CubieCube()
        last_axis = -1
        for _ in range(20):
            while True:
                mv = rng.randint(0, 17)
                axis = mv // 3
                if axis != last_axis:
                    break
            last_axis = axis
            power = mv % 3
            for _ in range(power + 1):
                cube.cornerMultiply(moveCube[axis])
                cube.edgeMultiply(moveCube[axis])
        try:
            fc_str = cube.toFaceCube().to_String()
            cubes.append(fc_str)
        except Exception:
            continue

    return cubes[:n]


def count_moves(solution: str) -> int:
    """Count moves in a solution string."""
    return len(solution.strip().split())


def benchmark_single(cube: str, max_phase1: int = 1, timeout: int = 5000,
                     time_budget_sec: float = None, use_neural: bool = False,
                     neural_strategy: str = 'none', neural_prob: float = 0.5,
                     neural_depth_cutoff: int = 5,
                     collect_stats: bool = False) -> dict:
    """Benchmark a single cube solve, optionally returning instrumentation."""
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
        result = {
            'success': True,
            'moves': moves,
            'time_ms': elapsed * 1000,
            'solution': solution,
        }
        if stats is not None:
            result['stats'] = stats
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time_ms': (time.time() - start) * 1000
        }


def run_benchmark(scrambles: list, max_phase1: int = 1, timeout: int = 5000,
                  time_budget_sec: float = None, use_neural: bool = False,
                  neural_strategy: str = 'none', neural_prob: float = 0.5,
                  neural_depth_cutoff: int = 5,
                  verbose: bool = False, label: str = "",
                  collect_stats: bool = False) -> dict:
    """Run benchmark on a list of scrambles."""
    results = []
    move_counts = []
    times = []
    all_nodes = []
    all_phase1_nodes = []
    all_phase2_nodes = []

    for i, cube in enumerate(scrambles):
        result = benchmark_single(cube, max_phase1, timeout, time_budget_sec,
                                  use_neural, neural_strategy, neural_prob,
                                  neural_depth_cutoff, collect_stats)
        results.append(result)

        if result['success']:
            move_counts.append(result['moves'])
            times.append(result['time_ms'])
            if 'stats' in result and result['stats']:
                all_nodes.append(result['stats']['nodes_expanded'])
                all_phase1_nodes.append(result['stats']['phase1_nodes'])
                all_phase2_nodes.append(result['stats']['phase2_nodes'])

        if verbose and (i + 1) % 10 == 0:
            print(f"  {label} Progress: {i + 1}/{len(scrambles)}", end='\r')

    if verbose:
        print()

    # Latency percentiles (p95, p99) for bounded-latency analysis
    sorted_times = sorted(times) if times else []
    n_t = len(sorted_times)
    p95_ms = sorted_times[int(0.95 * (n_t - 1))] if n_t > 0 else 0
    p99_ms = sorted_times[int(0.99 * (n_t - 1))] if n_t > 0 else 0

    summary = {
        'label': label,
        'max_phase1': max_phase1,
        'neural_strategy': neural_strategy,
        'total_scrambles': len(scrambles),
        'successful': len(move_counts),
        'move_stats': {
            'mean': statistics.mean(move_counts) if move_counts else 0,
            'median': statistics.median(move_counts) if move_counts else 0,
            'stdev': statistics.stdev(move_counts) if len(move_counts) > 1 else 0,
            'min': min(move_counts) if move_counts else 0,
            'max': max(move_counts) if move_counts else 0,
        },
        'time_stats': {
            'mean_ms': statistics.mean(times) if times else 0,
            'median_ms': statistics.median(times) if times else 0,
            'max_ms': max(times) if times else 0,
            'p95_ms': p95_ms,
            'p99_ms': p99_ms,
        },
        'distribution': {
            'optimal_20': sum(1 for m in move_counts if m <= 20),
            'near_optimal_18': sum(1 for m in move_counts if m <= 18),
            'good_21': sum(1 for m in move_counts if m <= 21),
        },
    }

    # Add node-expansion stats if available
    if all_nodes:
        summary['node_stats'] = {
            'mean_nodes': statistics.mean(all_nodes),
            'median_nodes': statistics.median(all_nodes),
            'mean_phase1_nodes': statistics.mean(all_phase1_nodes),
            'mean_phase2_nodes': statistics.mean(all_phase2_nodes),
        }

    # Keep per-cube move counts for paired statistical tests
    summary['_move_counts'] = move_counts
    summary['_node_counts'] = all_nodes

    return summary


# ─────────────────────────────────────────────────────────────────
# Statistical validation helpers
# ─────────────────────────────────────────────────────────────────

def paired_t_test(a: list, b: list):
    """
    Two-sided paired t-test on matched samples.

    Returns: (t_statistic, p_value, mean_diff, ci_lower, ci_upper)
    """
    assert len(a) == len(b), "Samples must be same length for paired test"
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    if n < 2:
        return (0.0, 1.0, mean_d, mean_d, mean_d)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t_stat = mean_d / se if se > 0 else 0.0
    # Approximate p-value using normal distribution (good for n >= 30)
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))
    # 95% CI for mean difference
    z95 = 1.96
    ci_lo = mean_d - z95 * se
    ci_hi = mean_d + z95 * se
    return (t_stat, p_value, mean_d, ci_lo, ci_hi)


def cohens_d(a: list, b: list) -> float:
    """Cohen's d effect size for paired samples."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    if len(diffs) < 2:
        return 0.0
    mean_d = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    return mean_d / sd if sd > 0 else 0.0


def bootstrap_ci(data: list, stat_fn=statistics.mean, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 42) -> tuple:
    """Bootstrap confidence interval for a statistic."""
    rng = random.Random(seed)
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    boot_stats = []
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    lo = boot_stats[int(alpha / 2 * n_boot)]
    hi = boot_stats[int((1 - alpha / 2) * n_boot)]
    return (stat_fn(data), lo, hi)


def print_results(results: dict):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {results['label']}")
    print(f"{'='*60}")
    strategy_info = f", neural_strategy={results['neural_strategy']}" if results.get('neural_strategy', 'none') != 'none' else ''
    print(f"Configuration: max_phase1_solutions = {results['max_phase1']}{strategy_info}")
    print(f"Scrambles: {results['successful']}/{results['total_scrambles']} successful")
    print()
    print("Move Statistics:")
    print(f"  Mean:   {results['move_stats']['mean']:.2f}")
    print(f"  Median: {results['move_stats']['median']:.1f}")
    print(f"  Stdev:  {results['move_stats']['stdev']:.2f}")
    print(f"  Range:  {results['move_stats']['min']} - {results['move_stats']['max']}")
    print()
    print("Solution Quality:")
    n = results['successful']
    if n > 0:
        print(f"  ≤18 moves (near-optimal): {results['distribution']['near_optimal_18']} ({100*results['distribution']['near_optimal_18']/n:.1f}%)")
        print(f"  ≤20 moves (optimal):      {results['distribution']['optimal_20']} ({100*results['distribution']['optimal_20']/n:.1f}%)")
        print(f"  ≤21 moves (good):         {results['distribution']['good_21']} ({100*results['distribution']['good_21']/n:.1f}%)")
    else:
        print("  (no successful solves)")
    print()
    print("Time Statistics:")
    print(f"  Mean:   {results['time_stats']['mean_ms']:.1f} ms")
    print(f"  Median: {results['time_stats']['median_ms']:.1f} ms")
    print(f"  P95:    {results['time_stats']['p95_ms']:.1f} ms")
    print(f"  P99:    {results['time_stats']['p99_ms']:.1f} ms")
    print(f"  Max:    {results['time_stats']['max_ms']:.1f} ms")
    if 'node_stats' in results:
        print()
        print("Node Expansion (instrumented):")
        ns = results['node_stats']
        print(f"  Mean total nodes:   {ns['mean_nodes']:.0f}")
        print(f"  Median total nodes: {ns['median_nodes']:.0f}")
        print(f"  Mean Phase-1 nodes: {ns['mean_phase1_nodes']:.0f}")
        print(f"  Mean Phase-2 nodes: {ns['mean_phase2_nodes']:.0f}")
    print(f"{'='*60}\n")


def compare_results(baseline: dict, improved: dict):
    """Compare baseline vs improved results with statistical tests."""
    print(f"\n{'='*70}")
    label = improved.get('label', 'Improved')
    print(f"COMPARISON: Baseline vs {label}")
    print(f"{'='*70}")

    b_mean = baseline['move_stats']['mean']
    i_mean = improved['move_stats']['mean']
    improvement = b_mean - i_mean
    pct = 100 * improvement / b_mean if b_mean > 0 else 0

    print(f"\nMove Count Improvement:")
    print(f"  Baseline:     {b_mean:.2f} moves (avg)")
    print(f"  {label}: {i_mean:.2f} moves (avg)")
    print(f"  Improvement:  {improvement:.2f} moves ({pct:.1f}%)")

    b_opt = baseline['distribution']['optimal_20']
    i_opt = improved['distribution']['optimal_20']
    n = baseline['successful']
    if n == 0:
        print("\nNo successful baseline solves to compare.")
        return

    print(f"\nOptimal Solutions (≤20 moves):")
    print(f"  Baseline:     {b_opt}/{n} ({100*b_opt/n:.1f}%)")
    print(f"  {label}: {i_opt}/{n} ({100*i_opt/n:.1f}%)")
    print(f"  Improvement:  +{i_opt - b_opt} ({100*(i_opt-b_opt)/n:.1f} pp)")

    b_time = baseline['time_stats']['mean_ms']
    i_time = improved['time_stats']['mean_ms']

    print(f"\nTime Trade-off:")
    print(f"  Baseline:     {b_time:.1f} ms (avg)")
    print(f"  {label}: {i_time:.1f} ms (avg)")
    if b_time > 0:
        print(f"  Overhead:     {i_time - b_time:.1f} ms ({100*(i_time-b_time)/b_time:.1f}% increase)")

    # Statistical validation (paired t-test)
    bm = baseline.get('_move_counts', [])
    im = improved.get('_move_counts', [])
    if len(bm) == len(im) and len(bm) >= 10:
        t_stat, p_val, mean_diff, ci_lo, ci_hi = paired_t_test(bm, im)
        d = cohens_d(bm, im)
        print(f"\nStatistical Validation (paired, n={len(bm)}):")
        print(f"  Mean diff:    {mean_diff:.3f} moves (baseline - improved)")
        print(f"  95% CI:       [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  t-statistic:  {t_stat:.3f}")
        print(f"  p-value:      {p_val:.4f}{'  ***' if p_val < 0.001 else '  **' if p_val < 0.01 else '  *' if p_val < 0.05 else '  (n.s.)'}")
        print(f"  Cohen's d:    {d:.3f}")

    # Node comparison
    bn = baseline.get('node_stats', {})
    in_ = improved.get('node_stats', {})
    if bn and in_:
        b_nodes = bn['mean_nodes']
        i_nodes = in_['mean_nodes']
        node_ratio = i_nodes / b_nodes if b_nodes > 0 else 0
        print(f"\nNode Expansion:")
        print(f"  Baseline:     {b_nodes:.0f} nodes (avg)")
        print(f"  {label}: {i_nodes:.0f} nodes (avg)")
        print(f"  Ratio:        {node_ratio:.2f}x")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark improved Kociemba algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Original multi-solution benchmark (backward compatible):
  python benchmark.py --scrambles 200 --compare-baseline --max-phase1 5 --seed 42 -v

  # Neural strategy comparison:
  python benchmark.py --scrambles 50 --neural-strategies none move_order probabilistic adaptive \\
      --max-phase1 5 --seed 42 --output-json results/neural_strategies.json -v

  # Hard-case benchmark:
  python benchmark.py --scrambles 50 --hard-cases --max-phase1 5 --seed 42 \\
      --output-json results/hard_cases.json -v

  # Anytime sweep:
  python benchmark.py --scrambles 200 --time-budgets 0.1 0.3 0.5 1.0 --max-phase1 5 --seed 42
""")
    parser.add_argument('--scrambles', type=int, default=100,
                        help='Number of random scrambles to test')
    parser.add_argument('--max-phase1', type=int, default=10,
                        help='max_phase1_solutions parameter for improved solver')
    parser.add_argument('--timeout', type=int, default=5000,
                        help='Timeout in milliseconds')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare against baseline (single solution)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--output-json', type=str, default=None, dest='output_json',
                        help='Alias for --output; write results to JSON for reproducibility')
    parser.add_argument('--time-budget', type=float, default=None,
                        help='Time budget in seconds for anytime mode (e.g. 0.5)')
    parser.add_argument('--time-budgets', type=float, nargs='*', default=None,
                        help='Sweep: run same scrambles for each time budget (e.g. 0.1 0.3 0.5 1.0)')
    parser.add_argument('--use-neural', action='store_true',
                        help='Use neural network for move ordering (requires trained model)')
    # ── New research flags ──
    parser.add_argument('--neural-strategy', type=str, default='none',
                        choices=['none', 'move_order', 'probabilistic', 'adaptive'],
                        help='Neural integration strategy for single run')
    parser.add_argument('--neural-strategies', type=str, nargs='*', default=None,
                        help='Compare multiple neural strategies (e.g. none move_order probabilistic adaptive)')
    parser.add_argument('--neural-prob', type=float, default=0.5,
                        help='Probability for probabilistic strategy (default: 0.5)')
    parser.add_argument('--neural-probs', type=float, nargs='*', default=None,
                        help='Sweep: try multiple probabilities for probabilistic strategy')
    parser.add_argument('--neural-depth-cutoff', type=int, default=5,
                        help='Depth cutoff for adaptive strategy (default: 5)')
    parser.add_argument('--hard-cases', action='store_true',
                        help='Generate hard cases instead of random scrambles')
    parser.add_argument('--collect-stats', action='store_true',
                        help='Collect node-expansion instrumentation (slower)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()
    output_file = args.output or args.output_json

    # Determine if neural is requested via strategy or flag
    use_neural = args.use_neural or (args.neural_strategy != 'none') or bool(args.neural_strategies)

    # ── Generate scrambles ──
    if args.hard_cases:
        print(f"\nGenerating {args.scrambles} hard-case cubes (seed={args.seed})...")
        scrambles = generate_hard_cases(args.scrambles, args.seed)
    else:
        print(f"\nGenerating {args.scrambles} random scrambles (seed={args.seed})...")
        scrambles = generate_scrambles(args.scrambles, args.seed)

    all_results = {}

    # ── Mode 1: time-budget sweep ──
    if args.time_budgets:
        all_results['time_budget_sweep'] = {}
        for t in args.time_budgets:
            label = f"Budget-{t}s"
            print(f"\nRunning anytime sweep: time_budget_sec={t} (max_phase1={args.max_phase1})...")
            res = run_benchmark(scrambles, max_phase1=args.max_phase1, timeout=args.timeout,
                               time_budget_sec=t, use_neural=use_neural,
                               neural_strategy=args.neural_strategy,
                               neural_prob=args.neural_prob,
                               neural_depth_cutoff=args.neural_depth_cutoff,
                               verbose=args.verbose, label=label,
                               collect_stats=args.collect_stats)
            print_results(res)
            n = res['successful']
            pct20 = (100 * res['distribution']['optimal_20'] / n) if n > 0 else 0
            entry = {
                'mean_moves': res['move_stats']['mean'],
                'median_moves': res['move_stats']['median'],
                'pct_le20': round(pct20, 1),
                'mean_time_ms': res['time_stats']['mean_ms'],
                'p95_time_ms': res['time_stats']['p95_ms'],
                'successful': n,
                'total': res['total_scrambles'],
            }
            if 'node_stats' in res:
                entry['node_stats'] = res['node_stats']
            all_results['time_budget_sweep'][str(t)] = entry

        all_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'scrambles': args.scrambles,
            'seed': args.seed,
            'max_phase1': args.max_phase1,
            'time_budgets': list(args.time_budgets),
            'use_neural': use_neural,
            'neural_strategy': args.neural_strategy,
            'hard_cases': args.hard_cases,
        }
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {output_file}")
        return

    # ── Mode 2: neural strategy comparison ──
    if args.neural_strategies:
        strategies = args.neural_strategies
        print(f"\n*** Neural Strategy Comparison ***")
        print(f"Strategies: {strategies}")
        print(f"max_phase1={args.max_phase1}, scrambles={len(scrambles)}")

        strategy_results = {}

        # Always run baseline (no neural, K=1) for reference
        print(f"\nRunning baseline (K=1, no neural)...")
        baseline = run_benchmark(scrambles, max_phase1=1, timeout=args.timeout,
                                time_budget_sec=args.time_budget,
                                use_neural=False, neural_strategy='none',
                                verbose=args.verbose, label="Baseline-K1",
                                collect_stats=True)
        print_results(baseline)
        strategy_results['baseline_k1'] = _clean_for_json(baseline)

        for strat in strategies:
            neural_on = (strat != 'none')
            label = f"K{args.max_phase1}-{strat}"
            print(f"\nRunning: {label}...")

            # For probabilistic strategy, sweep probabilities if requested
            if strat == 'probabilistic' and args.neural_probs:
                for p in args.neural_probs:
                    sub_label = f"K{args.max_phase1}-prob-{p}"
                    res = run_benchmark(scrambles, max_phase1=args.max_phase1,
                                       timeout=args.timeout,
                                       time_budget_sec=args.time_budget,
                                       use_neural=neural_on,
                                       neural_strategy=strat,
                                       neural_prob=p,
                                       neural_depth_cutoff=args.neural_depth_cutoff,
                                       verbose=args.verbose, label=sub_label,
                                       collect_stats=True)
                    print_results(res)
                    compare_results(baseline, res)
                    strategy_results[sub_label] = _clean_for_json(res)
            else:
                res = run_benchmark(scrambles, max_phase1=args.max_phase1,
                                   timeout=args.timeout,
                                   time_budget_sec=args.time_budget,
                                   use_neural=neural_on,
                                   neural_strategy=strat,
                                   neural_prob=args.neural_prob,
                                   neural_depth_cutoff=args.neural_depth_cutoff,
                                   verbose=args.verbose, label=label,
                                   collect_stats=True)
                print_results(res)
                compare_results(baseline, res)
                strategy_results[label] = _clean_for_json(res)

        all_results['strategy_comparison'] = strategy_results
        all_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'scrambles': args.scrambles,
            'seed': args.seed,
            'max_phase1': args.max_phase1,
            'timeout_ms': args.timeout,
            'time_budget_sec': args.time_budget,
            'strategies': strategies,
            'neural_prob': args.neural_prob,
            'neural_probs': args.neural_probs,
            'neural_depth_cutoff': args.neural_depth_cutoff,
            'hard_cases': args.hard_cases,
        }
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {output_file}")
        return

    # ── Mode 3: original compare-baseline mode (backward compatible) ──
    if args.compare_baseline:
        print(f"\nRunning baseline (max_phase1_solutions=1)...")
        baseline = run_benchmark(scrambles, max_phase1=1, timeout=args.timeout,
                                time_budget_sec=args.time_budget,
                                use_neural=False, neural_strategy='none',
                                verbose=args.verbose, label="Baseline",
                                collect_stats=args.collect_stats)
        print_results(baseline)
        all_results['baseline'] = _clean_for_json(baseline)

    ns = args.neural_strategy if args.neural_strategy != 'none' else 'none'
    label_suffix = f"-{ns}" if ns != 'none' else ''
    print(f"\nRunning improved solver (max_phase1_solutions={args.max_phase1}{', strategy=' + ns if ns != 'none' else ''})...")
    improved = run_benchmark(scrambles, max_phase1=args.max_phase1, timeout=args.timeout,
                            time_budget_sec=args.time_budget,
                            use_neural=use_neural, neural_strategy=ns,
                            neural_prob=args.neural_prob,
                            neural_depth_cutoff=args.neural_depth_cutoff,
                            verbose=args.verbose,
                            label=f"Multi-{args.max_phase1}{label_suffix}",
                            collect_stats=args.collect_stats)
    print_results(improved)
    all_results['improved'] = _clean_for_json(improved)

    if args.compare_baseline:
        compare_results(baseline, improved)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        all_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'scrambles': args.scrambles,
            'seed': args.seed,
            'timeout_ms': args.timeout,
            'time_budget_sec': args.time_budget,
            'use_neural': use_neural,
            'neural_strategy': ns,
            'neural_prob': args.neural_prob,
            'neural_depth_cutoff': args.neural_depth_cutoff,
            'hard_cases': args.hard_cases,
        }
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_file}")


def _clean_for_json(results: dict) -> dict:
    """Remove internal keys (prefixed with _) before serialising to JSON."""
    return {k: v for k, v in results.items() if not k.startswith('_')}


if __name__ == '__main__':
    main()
