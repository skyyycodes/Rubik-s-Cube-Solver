#!/usr/bin/env python3
"""200-cube benchmark, seed 42, comparing Baseline / K=5 / LUT (retrained model)."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kociemba'))

import importlib
import kociemba.neural_heuristic as nh
importlib.reload(nh)
import kociemba.pykociemba.search as search_mod
importlib.reload(search_mod)
search_mod._neural_move_lut = None

from kociemba.pykociemba import tools
from kociemba import _solve_python_instrumented

import numpy as np
import random
np.random.seed(42)
random.seed(42)  # tools.randomCube uses stdlib random, not numpy

n_cubes = 200
cubes = [tools.randomCube() for _ in range(n_cubes)]
print(f'Generated {n_cubes} cubes (seed 42)', flush=True)

def run(name, kwargs):
    print(f'\n=== {name} ===', flush=True)
    moves, nodes, times = [], [], []
    for i, cube in enumerate(cubes):
        t0 = time.time()
        try:
            sol, stats = _solve_python_instrumented(cube, None, **kwargs)
            ms = (time.time() - t0) * 1000
            moves.append(len(sol.split()))
            nodes.append(stats['nodes_expanded'])
            times.append(ms)
        except Exception as e:
            ms = (time.time() - t0) * 1000
            moves.append(None)
            nodes.append(None)
            times.append(ms)
        if (i+1) % 25 == 0:
            print(f'  {i+1}/{n_cubes}', flush=True)
    return moves, nodes, times

bl_m, bl_n, bl_t = run('Baseline K=1', dict(max_phase1_solutions=1, timeout=5000))
k5_m, k5_n, k5_t = run('K=5 no neural', dict(max_phase1_solutions=5, timeout=5000))
lut_m, lut_n, lut_t = run('K=5 + LUT (retrained)', dict(max_phase1_solutions=5, timeout=5000, neural_strategy='lut'))

def stats(name, moves, nodes, times):
    solved = [m for m in moves if m is not None]
    n_ok = [n for n in nodes if n is not None]
    pct20 = sum(1 for m in solved if m <= 20) / len(solved) * 100 if solved else 0
    return {
        'name': name,
        'solved': len(solved),
        'mean_moves': np.mean(solved) if solved else 0,
        'median_moves': int(np.median(solved)) if solved else 0,
        'pct20': pct20,
        'mean_nodes': np.mean(n_ok) if n_ok else 0,
        'mean_time': np.mean(times),
    }

results = [
    stats('Baseline K=1', bl_m, bl_n, bl_t),
    stats('K=5 no neural', k5_m, k5_n, k5_t),
    stats('K=5 + LUT', lut_m, lut_n, lut_t),
]

print('\n' + '='*70, flush=True)
print('FINAL RESULTS (200 cubes, seed 42)', flush=True)
print('='*70, flush=True)
print(f'{"Strategy":18s} {"Solved":>7s} {"Mean":>7s} {"Med":>5s} {"≤20%":>7s} {"Nodes":>12s} {"Time(ms)":>10s}', flush=True)
for r in results:
    print(f'{r["name"]:18s} {r["solved"]:>7d} {r["mean_moves"]:>7.2f} {r["median_moves"]:>5d} {r["pct20"]:>6.1f}% {r["mean_nodes"]:>12,.0f} {r["mean_time"]:>10.0f}', flush=True)

if results[1]['mean_nodes'] and results[2]['mean_nodes']:
    ratio = results[1]['mean_nodes'] / results[2]['mean_nodes']
    print(f'\nNode reduction (K=5 → LUT): {ratio:.2f}x', flush=True)

# Save results to file
import json
with open('benchmark_results.json', 'w') as f:
    json.dump({
        'baseline': {'moves': bl_m, 'nodes': bl_n, 'times': bl_t},
        'k5': {'moves': k5_m, 'nodes': k5_n, 'times': k5_t},
        'lut': {'moves': lut_m, 'nodes': lut_n, 'times': lut_t},
        'summary': results,
    }, f, indent=2)
print('\nSaved to benchmark_results.json', flush=True)
