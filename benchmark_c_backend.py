#!/usr/bin/env python3
"""
Benchmark the C backend solver on the same scrambles used in the main benchmark.
Generates timing and solution-quality data for the C backend as an external baseline.
"""
import sys
import time
import json
import statistics
import random

sys.path.insert(0, 'kociemba')

from kociemba import _c_solver_available, _solve_c, _solve_python
from kociemba.pykociemba.tools import randomCube


def generate_scrambles(n: int, seed: int = 42) -> list:
    """Generate n random scrambles using the same method as benchmark.py.
    Uses global random.seed() to match benchmark.py's behavior exactly."""
    random.seed(seed)
    return [randomCube() for _ in range(n)]


def count_moves(solution: str) -> int:
    return len(solution.strip().split())


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark C backend solver')
    parser.add_argument('--scrambles', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-depth', type=int, default=24)
    parser.add_argument('--output-json', type=str, default='results/c_backend_benchmark.json')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if not _c_solver_available:
        print("ERROR: C backend not available. Cannot run this benchmark.")
        sys.exit(1)

    print(f"C backend available: {_c_solver_available}")
    print(f"Generating {args.scrambles} random scrambles (seed={args.seed})...")

    # Use the SAME scramble generation as benchmark.py
    # benchmark.py uses generate_scrambles which calls randomCube()
    # We need to match the seed behavior exactly
    scrambles = generate_scrambles(args.scrambles, args.seed)
    print(f"Generated {len(scrambles)} scrambles.\n")

    # ── C backend benchmark ──
    print("=" * 70)
    print("C Backend (single solution, K=1)")
    print("=" * 70)
    c_results = []
    c_moves = []
    c_times = []
    c_failures = 0

    for i, cube in enumerate(scrambles):
        start = time.time()
        try:
            solution = _solve_c(cube, None, args.max_depth)
            elapsed = time.time() - start
            moves = count_moves(solution)
            c_results.append({
                'success': True,
                'moves': moves,
                'time_ms': elapsed * 1000,
                'solution': solution
            })
            c_moves.append(moves)
            c_times.append(elapsed * 1000)
        except Exception as e:
            elapsed = time.time() - start
            c_results.append({
                'success': False,
                'error': str(e),
                'time_ms': elapsed * 1000
            })
            c_failures += 1

        if args.verbose and (i + 1) % 20 == 0:
            print(f"  C backend progress: {i + 1}/{len(scrambles)}")

    # ── Python backend K=1 (for direct comparison) ──
    print("\n" + "=" * 70)
    print("Python Backend (single solution, K=1)")
    print("=" * 70)
    py_results = []
    py_moves = []
    py_times = []
    py_failures = 0

    for i, cube in enumerate(scrambles):
        start = time.time()
        try:
            solution = _solve_python(cube, None, args.max_depth, max_phase1_solutions=1,
                                     timeout=5000)
            elapsed = time.time() - start
            moves = count_moves(solution)
            py_results.append({
                'success': True,
                'moves': moves,
                'time_ms': elapsed * 1000,
                'solution': solution
            })
            py_moves.append(moves)
            py_times.append(elapsed * 1000)
        except Exception as e:
            elapsed = time.time() - start
            py_results.append({
                'success': False,
                'error': str(e),
                'time_ms': elapsed * 1000
            })
            py_failures += 1

        if args.verbose and (i + 1) % 20 == 0:
            print(f"  Python K=1 progress: {i + 1}/{len(scrambles)}")

    # ── Python backend K=5 (for comparison) ──
    print("\n" + "=" * 70)
    print("Python Backend (multi-solution, K=5)")
    print("=" * 70)
    py5_results = []
    py5_moves = []
    py5_times = []
    py5_failures = 0

    for i, cube in enumerate(scrambles):
        start = time.time()
        try:
            solution = _solve_python(cube, None, args.max_depth, max_phase1_solutions=5,
                                     timeout=5000)
            elapsed = time.time() - start
            moves = count_moves(solution)
            py5_results.append({
                'success': True,
                'moves': moves,
                'time_ms': elapsed * 1000,
                'solution': solution
            })
            py5_moves.append(moves)
            py5_times.append(elapsed * 1000)
        except Exception as e:
            elapsed = time.time() - start
            py5_results.append({
                'success': False,
                'error': str(e),
                'time_ms': elapsed * 1000
            })
            py5_failures += 1

        if args.verbose and (i + 1) % 20 == 0:
            print(f"  Python K=5 progress: {i + 1}/{len(scrambles)}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    def summarize(label, moves_list, times_list, failures):
        n = len(moves_list)
        le20 = sum(1 for m in moves_list if m <= 20)
        pct20 = 100 * le20 / n if n > 0 else 0
        sorted_t = sorted(times_list)
        p95 = sorted_t[int(0.95 * (n - 1))] if n > 0 else 0
        p99 = sorted_t[int(0.99 * (n - 1))] if n > 0 else 0
        print(f"\n{label}:")
        print(f"  Successful: {n}/{n + failures}")
        print(f"  Mean moves: {statistics.mean(moves_list):.2f}")
        print(f"  Median moves: {statistics.median(moves_list):.1f}")
        print(f"  Stdev moves: {statistics.stdev(moves_list):.2f}" if n > 1 else "")
        print(f"  ≤20 moves: {le20}/{n} ({pct20:.1f}%)")
        print(f"  Mean time: {statistics.mean(times_list):.2f} ms")
        print(f"  Median time: {statistics.median(times_list):.2f} ms")
        print(f"  P95 time: {p95:.2f} ms")
        print(f"  P99 time: {p99:.2f} ms")
        print(f"  Max time: {max(times_list):.2f} ms")
        print(f"  Min time: {min(times_list):.2f} ms")
        return {
            'successful': n,
            'failures': failures,
            'mean_moves': round(statistics.mean(moves_list), 2),
            'median_moves': statistics.median(moves_list),
            'stdev_moves': round(statistics.stdev(moves_list), 2) if n > 1 else 0,
            'le20_count': le20,
            'le20_pct': round(pct20, 1),
            'mean_time_ms': round(statistics.mean(times_list), 2),
            'median_time_ms': round(statistics.median(times_list), 2),
            'p95_time_ms': round(p95, 2),
            'p99_time_ms': round(p99, 2),
            'max_time_ms': round(max(times_list), 2),
            'min_time_ms': round(min(times_list), 2),
        }

    c_summary = summarize("C Backend (K=1)", c_moves, c_times, c_failures)
    py_summary = summarize("Python Backend (K=1)", py_moves, py_times, py_failures)
    py5_summary = summarize("Python Backend (K=5)", py5_moves, py5_times, py5_failures)

    # Speedup
    if c_times and py_times:
        speedup = statistics.mean(py_times) / statistics.mean(c_times)
        print(f"\n  C vs Python K=1 speedup: {speedup:.1f}x")
    if c_times and py5_times:
        speedup5 = statistics.mean(py5_times) / statistics.mean(c_times)
        print(f"  C vs Python K=5 speedup: {speedup5:.1f}x")

    # Check if C and Python K=1 produce same moves (they should, same algorithm)
    if c_moves and py_moves and len(c_moves) == len(py_moves):
        same = sum(1 for a, b in zip(c_moves, py_moves) if a == b)
        print(f"\n  C vs Python K=1 same move count: {same}/{len(c_moves)} ({100*same/len(c_moves):.1f}%)")
        mean_diff = statistics.mean([a - b for a, b in zip(c_moves, py_moves)])
        print(f"  Mean move difference (C - Py): {mean_diff:.3f}")

    # ── Save results ──
    output = {
        'metadata': {
            'scrambles': args.scrambles,
            'seed': args.seed,
            'max_depth': args.max_depth,
        },
        'c_backend_k1': c_summary,
        'python_k1': py_summary,
        'python_k5': py5_summary,
    }

    from pathlib import Path
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
