"""C-backend benchmark: baseline solve() vs solve_hsort() on 50 random cubes.

Measures Phase-1 node expansions and wall-clock latency for both, using the
native backend. This is the complement to the Python-backend h_sort experiment
in results/h_sort_benchmark.json: in Python the 1.66x node reduction is eaten
by per-lookup overhead; in C each getPruning call is a single table load, so
we expect the node reduction to translate to wall-clock speedup.
"""
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, 'kociemba')
import kociemba
from kociemba.ckociembawrapper import ffi, lib
from kociemba.pykociemba import tools

CACHE_DIR = os.path.join(os.path.dirname(kociemba.__file__), 'cprunetables').encode()


def run_one(cube, fn, *extra):
    t0 = time.perf_counter()
    r = fn(cube.encode(), ffi.NULL, CACHE_DIR, 24, *extra)
    t1 = time.perf_counter()
    if r == ffi.NULL:
        return None, None, None
    sol = ffi.string(r).decode().strip()
    moves = len([t for t in sol.split() if t])
    return sol, moves, (t1 - t0) * 1000.0, lib.get_phase1_nodes()


def summarize(name, results):
    moves = [r['moves'] for r in results]
    times = [r['time_ms'] for r in results]
    nodes = [r['nodes'] for r in results]
    print(f"\n=== {name} ===")
    print(f"  n successful:  {len(results)}")
    print(f"  moves   mean={statistics.mean(moves):.2f}  median={statistics.median(moves):.1f}  "
          f"range={min(moves)}-{max(moves)}")
    print(f"  time_ms mean={statistics.mean(times):.3f}  median={statistics.median(times):.3f}  "
          f"max={max(times):.3f}")
    print(f"  nodes   mean={statistics.mean(nodes):.1f}  median={statistics.median(nodes):.1f}  "
          f"max={max(nodes)}")
    near = sum(1 for m in moves if m <= 20)
    print(f"  <=20 moves: {near}/{len(results)} ({100*near/len(results):.1f}%)")


def main():
    random.seed(42)
    n = 50
    scrambles = [tools.randomCube() for _ in range(n)]

    # warmup (table mmap, first solve loads prunetables)
    _ = ffi.string(lib.solve(scrambles[0].encode(), ffi.NULL, CACHE_DIR, 24)).decode()
    _ = ffi.string(lib.solve_hsort(scrambles[0].encode(), ffi.NULL, CACHE_DIR, 24, 30)).decode()

    baseline_results = []
    hsort_results = []

    for i, cube in enumerate(scrambles):
        t0 = time.perf_counter()
        r = lib.solve(cube.encode(), ffi.NULL, CACHE_DIR, 24)
        t1 = time.perf_counter()
        nodes_b = lib.get_phase1_nodes()
        if r == ffi.NULL:
            print(f"  [{i}] baseline FAILED")
            continue
        sol_b = ffi.string(r).decode().strip()
        moves_b = len(sol_b.split())
        baseline_results.append({
            'cube': cube, 'solution': sol_b, 'moves': moves_b,
            'time_ms': (t1 - t0) * 1000.0, 'nodes': nodes_b,
        })

        t0 = time.perf_counter()
        r = lib.solve_hsort(cube.encode(), ffi.NULL, CACHE_DIR, 24, 30)
        t1 = time.perf_counter()
        nodes_h = lib.get_phase1_nodes()
        if r == ffi.NULL:
            print(f"  [{i}] hsort FAILED")
            continue
        sol_h = ffi.string(r).decode().strip()
        moves_h = len(sol_h.split())
        hsort_results.append({
            'cube': cube, 'solution': sol_h, 'moves': moves_h,
            'time_ms': (t1 - t0) * 1000.0, 'nodes': nodes_h,
        })
        print(f"  [{i:2}] base: {moves_b:2} moves {(baseline_results[-1]['time_ms']):7.2f}ms "
              f"{nodes_b:7d} nodes   | hsort: {moves_h:2} moves "
              f"{(hsort_results[-1]['time_ms']):7.2f}ms {nodes_h:7d} nodes")

    summarize('C baseline (first-solution IDA*)', baseline_results)
    summarize('C h_sort   (oracle move ordering)', hsort_results)

    # Paired comparison
    assert len(baseline_results) == len(hsort_results)
    node_ratios = []
    time_ratios = []
    for b, h in zip(baseline_results, hsort_results):
        if h['nodes'] > 0:
            node_ratios.append(b['nodes'] / h['nodes'])
        if h['time_ms'] > 0:
            time_ratios.append(b['time_ms'] / h['time_ms'])
    print("\n=== Paired ratios (baseline / hsort) ===")
    print(f"  node ratio   mean={statistics.mean(node_ratios):.2f}x  "
          f"median={statistics.median(node_ratios):.2f}x  "
          f"min={min(node_ratios):.2f}x  max={max(node_ratios):.2f}x")
    print(f"  time ratio   mean={statistics.mean(time_ratios):.2f}x  "
          f"median={statistics.median(time_ratios):.2f}x  "
          f"min={min(time_ratios):.2f}x  max={max(time_ratios):.2f}x")

    out = {
        'seed': 42,
        'n': n,
        'baseline': baseline_results,
        'hsort': hsort_results,
        'summary': {
            'baseline_mean_nodes': statistics.mean(r['nodes'] for r in baseline_results),
            'hsort_mean_nodes': statistics.mean(r['nodes'] for r in hsort_results),
            'baseline_mean_time_ms': statistics.mean(r['time_ms'] for r in baseline_results),
            'hsort_mean_time_ms': statistics.mean(r['time_ms'] for r in hsort_results),
            'node_ratio_mean': statistics.mean(node_ratios),
            'time_ratio_mean': statistics.mean(time_ratios),
            'baseline_mean_moves': statistics.mean(r['moves'] for r in baseline_results),
            'hsort_mean_moves': statistics.mean(r['moves'] for r in hsort_results),
        },
    }
    Path('results').mkdir(exist_ok=True)
    with open('results/hsort_c_backend.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/hsort_c_backend.json")


if __name__ == '__main__':
    main()
