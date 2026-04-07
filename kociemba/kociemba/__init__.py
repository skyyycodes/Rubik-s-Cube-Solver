import warnings


class SlowContextWarning(Warning):
    pass


# Error messages for both implementations
_errors = {
    'Error 1': 'There is not exactly one facelet of each colour',
    'Error 2': 'Not all 12 edges exist exactly once',
    'Error 3': 'Flip error: One edge has to be flipped',
    'Error 4': 'Not all corners exist exactly once',
    'Error 5': 'Twist error: One corner has to be twisted',
    'Error 6': 'Parity error: Two corners or two edges have to be exchanged',
    'Error 7': 'No solution exists for the given maxDepth',
    'Error 8': 'Timeout, no solution within given time'
}

# Try to load C implementation
_c_solver_available = False
try:
    import os
    from .ckociembawrapper import ffi, lib
    cache_dir = os.path.join(os.path.dirname(__file__), 'cprunetables')
    _c_solver_available = True
except ImportError as e:
    pass

# Always load Python implementation for multi-solution search
from .pykociemba import search


def _solve_python(cube, pattern, max_depth=24, max_phase1_solutions=1, timeout=1000,
                  time_budget_sec=None, use_neural=False, neural_strategy='none',
                  neural_prob=0.5, neural_depth_cutoff=5):
    """Python implementation - supports multi-solution Phase 1 search, time-bounded anytime, neural strategies."""
    if pattern is not None:
        cube = search.patternize(cube, pattern)
    # Time limit in seconds: explicit time_budget_sec takes precedence; else timeout is in ms
    timeout_sec = time_budget_sec if time_budget_sec is not None else (timeout / 1000.0)
    searcher = search.Search()
    res = searcher.solution(cube, max_depth, timeout_sec, False, max_phase1_solutions,
                            use_neural, neural_strategy, neural_prob,
                            neural_depth_cutoff).strip()
    if res in _errors:
        raise ValueError(_errors[res])
    return res


def _solve_python_instrumented(cube, pattern, max_depth=24, max_phase1_solutions=1,
                               timeout=1000, time_budget_sec=None, use_neural=False,
                               neural_strategy='none', neural_prob=0.5,
                               neural_depth_cutoff=5):
    """Like _solve_python but also returns instrumentation stats.

    Returns (solution_string, stats_dict).
    """
    if pattern is not None:
        cube = search.patternize(cube, pattern)
    timeout_sec = time_budget_sec if time_budget_sec is not None else (timeout / 1000.0)
    searcher = search.Search()
    res = searcher.solution(cube, max_depth, timeout_sec, False, max_phase1_solutions,
                            use_neural, neural_strategy, neural_prob,
                            neural_depth_cutoff).strip()
    stats = searcher.get_stats()
    if res in _errors:
        raise ValueError(_errors[res])
    return res, stats


def _solve_c(cube, pattern, max_depth=24):
    """C implementation - fast but single solution only."""
    pattern_utf8 = pattern.encode('utf-8') if pattern is not None else ffi.NULL
    res = lib.solve(cube.encode('utf-8'), pattern_utf8, cache_dir.encode('utf-8'), max_depth)
    if res != ffi.NULL:
        return ffi.string(res).strip().decode('utf-8')
    else:
        raise ValueError('Error. Probably cubestring is invalid')


def solve(cubestring, patternstring=None, max_depth=24, max_phase1_solutions=1,
          timeout=1000, time_budget_sec=None, use_neural=False,
          neural_strategy='none', neural_prob=0.5, neural_depth_cutoff=5):
    """
    Solve a Rubik's cube using two-phase algorithm.

    @param cubestring: 54-character string representing cube state (URFDLB order)
    @param patternstring: Optional target pattern to solve toward
    @param max_depth: Maximum solution length (default: 24)
    @param max_phase1_solutions: Number of Phase 1 solutions to explore for better results.
                                 Higher values find shorter solutions but take more time.
                                 Default: 1 (original behavior), Recommended: 5-10 for research.
    @param timeout: Maximum time in milliseconds (default: 1000). Ignored if time_budget_sec is set.
    @param time_budget_sec: Optional time budget in seconds (e.g. 0.5 for 500 ms). When set, the
                            solver returns the best solution found so far when the budget is exceeded
                            (anytime behavior). None = no limit (use timeout instead).
    @param use_neural: Use neural network for move ordering. Requires trained model.
                       Default: False.
    @param neural_strategy: Neural integration strategy:
                            'none'          - no neural ordering (baseline)
                            'move_order'    - always reorder moves by neural prediction (A)
                            'probabilistic' - use neural with probability neural_prob (B)
                            'adaptive'      - neural at shallow depths only (C)
    @param neural_prob: Probability for probabilistic strategy (default: 0.5).
    @param neural_depth_cutoff: Depth threshold for adaptive strategy (default: 5).

    >>> solve("BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF")
    "B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'"

    # Multi-solution search for shorter solutions (research improvement):
    >>> solve("...", max_phase1_solutions=10)
    # Time-bounded anytime (return best solution within 500 ms):
    >>> solve("...", max_phase1_solutions=20, time_budget_sec=0.5)
    # Neural strategies:
    >>> solve("...", use_neural=True, neural_strategy='move_order')
    >>> solve("...", use_neural=True, neural_strategy='probabilistic', neural_prob=0.3)
    >>> solve("...", use_neural=True, neural_strategy='adaptive', neural_depth_cutoff=5)
    """

    # Use C solver for single-solution mode (fast), Python for enhanced features
    if (max_phase1_solutions == 1 and not use_neural and time_budget_sec is None
            and neural_strategy == 'none' and _c_solver_available):
        return _solve_c(cubestring, patternstring, max_depth)
    else:
        return _solve_python(cubestring, patternstring, max_depth, max_phase1_solutions,
                            timeout, time_budget_sec, use_neural, neural_strategy,
                            neural_prob, neural_depth_cutoff)


__all__ = ['solve']
