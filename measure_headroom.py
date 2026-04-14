"""
Empirical measurement of Kociemba Phase 1 move-ordering headroom.

Question we are answering:
    At a typical Phase 1 search state, how much information does "pick the
    child with smallest h" carry relative to the fixed lex-order baseline, and
    how much headroom is left for any learned move-ordering policy to improve?

Because the pruning-table heuristic is consistent (h(child) >= h(state) - 1
for all moves), each legal child falls into one of three buckets:
    - progressing: h(child) = h(state) - 1
    - plateau:     h(child) = h(state)
    - worse:       h(child) = h(state) + 1
The distribution over these buckets determines:
  (a) q_prog = P(a random legal child is progressing) -- the baseline hit rate
  (b) p_top  = P(min-h child is progressing) -- the pruning-table-sorted rate
  (c) p_lex  = P(first lex-order child is progressing) -- the Kociemba-default
              baseline
The neural-ordering headroom is upper-bounded by  p_top - p_lex : no policy
that assigns a scalar score per child can do better than putting the min-h
child first. When this gap is small, Observation 3's node-reduction ceiling
(1 - (p - p_0)(1 - 1/b))^d collapses to ~1x.

Sampling strategy: to avoid the bias of greedy descent (which stalls at states
where no progressing child exists), we do a UNIFORM RANDOM WALK of length
`walk_len` in the Phase 1 move space starting from a uniformly random cube.
The consecutive-move filter is applied. At each visited state we record the
child-bucket histogram.

Output: a JSON at results/headroom_measurement.json.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kociemba"))

from kociemba.pykociemba.cubiecube import CubieCube
from kociemba.pykociemba.coordcube import CoordCube, getPruning


N_SLICE1 = CoordCube.N_SLICE1


def h_phase1(twist, flip, slice_):
    """Phase 1 admissible heuristic: max over the two pruning tables."""
    return max(
        getPruning(CoordCube.Slice_Flip_Prun, N_SLICE1 * flip + slice_),
        getPruning(CoordCube.Slice_Twist_Prun, N_SLICE1 * twist + slice_),
    )


def make_random_coord_cube(rng):
    """Generate a random Phase 1 state via a uniformly-random CubieCube."""
    while True:
        cc = CubieCube()
        cc.setFlip(rng.randint(0, CoordCube.N_FLIP - 1))
        cc.setTwist(rng.randint(0, CoordCube.N_TWIST - 1))
        cc.setURFtoDLB(rng.randint(0, CoordCube.N_URFtoDLB - 1))
        cc.setURtoBR(rng.randint(0, CoordCube.N_URtoBR - 1))
        if (cc.edgeParity() ^ cc.cornerParity()) == 0:
            return CoordCube(cc)


def legal_children(twist, flip, FRtoBR, parent_ax):
    """
    Yield (move_idx, ax, po, twist', flip', slice', h') for each legal move,
    applying Kociemba's consecutive-move filter: ax != parent_ax and
    ax != parent_ax - 3.
    """
    out = []
    for ax in range(6):
        if parent_ax is not None and (ax == parent_ax or ax == parent_ax - 3):
            continue
        for po in range(1, 4):
            m = 3 * ax + (po - 1)
            new_twist = CoordCube.twistMove[twist][m]
            new_flip = CoordCube.flipMove[flip][m]
            new_FRtoBR = CoordCube.FRtoBR_Move[FRtoBR][m]
            new_slice = new_FRtoBR // 24
            hc = h_phase1(new_twist, new_flip, new_slice)
            out.append((m, ax, po, new_twist, new_flip, new_FRtoBR, new_slice, hc))
    return out


def walk(cube, rng, walk_len=30):
    """
    Uniform random walk in the Phase 1 move space. At each visited state,
    record full child-bucket stats, then take a uniformly random legal child
    (consecutive-move filter applied).
    """
    twist = cube.twist
    flip = cube.flip
    FRtoBR = cube.FRtoBR
    slice_ = FRtoBR // 24
    parent_ax = None

    steps = []
    for step in range(walk_len):
        h = h_phase1(twist, flip, slice_)
        if h == 0:
            break
        children = legal_children(twist, flip, FRtoBR, parent_ax)
        if not children:
            break

        child_hs = [c[-1] for c in children]
        n_legal = len(children)
        n_progressing = sum(1 for hc in child_hs if hc == h - 1)
        n_plateau = sum(1 for hc in child_hs if hc == h)
        n_worse = sum(1 for hc in child_hs if hc == h + 1)
        min_h = min(child_hs)

        # p_top: sort by h ascending; the *first* child in that sort is the
        # neural-predictor's best-case target. Count it as "progressing" iff
        # its h equals h-1.
        top_is_progressing = (min_h == h - 1)
        # p_lex: fixed lex order (move index 0,1,2,...). Count whether the
        # *first legal* lex-ordered child is progressing.
        # children list is already in lex order by construction of
        # legal_children (ax asc, po asc), so children[0] is the lex-first
        # legal move.
        lex_first_h = children[0][-1]
        lex_is_progressing = (lex_first_h == h - 1)

        steps.append({
            "h": h,
            "depth": step,
            "n_legal": n_legal,
            "n_progressing": n_progressing,
            "n_plateau": n_plateau,
            "n_worse": n_worse,
            "min_child_h": min_h,
            "top_is_progressing": top_is_progressing,
            "lex_is_progressing": lex_is_progressing,
        })

        chosen = rng.choice(children)
        _, ax, _, twist, flip, FRtoBR, slice_, _ = chosen
        parent_ax = ax

    return steps


def aggregate(all_steps):
    n_states = len(all_steps)
    if n_states == 0:
        return {}

    p_top = sum(s["top_is_progressing"] for s in all_steps) / n_states
    p_lex = sum(s["lex_is_progressing"] for s in all_steps) / n_states

    mean_legal = sum(s["n_legal"] for s in all_steps) / n_states
    mean_progressing = sum(s["n_progressing"] for s in all_steps) / n_states
    mean_plateau = sum(s["n_plateau"] for s in all_steps) / n_states
    mean_worse = sum(s["n_worse"] for s in all_steps) / n_states
    frac_progressing = mean_progressing / mean_legal if mean_legal > 0 else 0.0

    by_h = defaultdict(list)
    for s in all_steps:
        by_h[s["h"]].append(s)

    per_h = {}
    for h_val, rows in sorted(by_h.items()):
        if not rows:
            continue
        n = len(rows)
        per_h[h_val] = {
            "n_states": n,
            "p_top": sum(r["top_is_progressing"] for r in rows) / n,
            "p_lex": sum(r["lex_is_progressing"] for r in rows) / n,
            "mean_legal": sum(r["n_legal"] for r in rows) / n,
            "mean_progressing": sum(r["n_progressing"] for r in rows) / n,
            "frac_progressing": (sum(r["n_progressing"] for r in rows)
                                 / sum(r["n_legal"] for r in rows)),
        }

    return {
        "n_states": n_states,
        "p_top": p_top,
        "p_lex": p_lex,
        "headroom": p_top - p_lex,
        "mean_legal_children": mean_legal,
        "mean_progressing_children": mean_progressing,
        "mean_plateau_children": mean_plateau,
        "mean_worse_children": mean_worse,
        "frac_progressing_children": frac_progressing,
        "per_h": per_h,
    }


def main():
    rng = random.Random(42)
    n_cubes = 200
    walk_len = 30

    all_steps = []
    for i in range(n_cubes):
        cube = make_random_coord_cube(rng)
        steps = walk(cube, rng, walk_len=walk_len)
        all_steps.extend(steps)
        if (i + 1) % 50 == 0:
            print(f"  cube {i + 1}/{n_cubes}: total {len(all_steps)} states",
                  flush=True)

    summary = aggregate(all_steps)
    summary["config"] = {"n_cubes": n_cubes, "walk_len": walk_len, "seed": 42}

    out_path = Path(__file__).resolve().parent / "results" / "headroom_measurement.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 68)
    print(f"HEADROOM MEASUREMENT (n={n_cubes} cubes, {summary['n_states']} states)")
    print("=" * 68)
    print(f"  p_top  (min-h child = h-1, ORACLE sort):  {summary['p_top']:.4f}")
    print(f"  p_lex  (first-lex child = h-1, BASELINE): {summary['p_lex']:.4f}")
    print(f"  headroom (p_top - p_lex):                 {summary['headroom']:.4f}")
    print()
    print(f"  mean legal children / state:              {summary['mean_legal_children']:.2f}")
    print(f"  mean progressing (h-1) / state:           {summary['mean_progressing_children']:.2f}")
    print(f"  mean plateau     (h)   / state:           {summary['mean_plateau_children']:.2f}")
    print(f"  mean worse       (h+1) / state:           {summary['mean_worse_children']:.2f}")
    print(f"  fraction of legal that progress:          {summary['frac_progressing_children']:.4f}")
    print()
    print("  per-h breakdown:")
    print(f"    {'h':>3}  {'n':>5}  {'p_top':>6}  {'p_lex':>6}  "
          f"{'mean_prog':>9}  {'frac_prog':>9}")
    for h_val, row in sorted(summary["per_h"].items(), reverse=True):
        print(f"    {h_val:>3}  {row['n_states']:>5}  "
              f"{row['p_top']:>6.3f}  {row['p_lex']:>6.3f}  "
              f"{row['mean_progressing']:>9.2f}  {row['frac_progressing']:>9.3f}")
    print()
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
