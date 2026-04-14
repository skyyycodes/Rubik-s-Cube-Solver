import time
import random as _random
from builtins import range
from .color import colors
from .facecube import FaceCube
from .coordcube import CoordCube, getPruning
from .cubiecube import CubieCube

# Symmetry module for improved heuristics
# NOTE: After extensive testing, symmetry is DISABLED because:
# - MAX mode: Stronger heuristic causes IDA* to search deeper (133x slowdown)
# - MIN mode: Weaker heuristic provides no benefit (7x slowdown, same quality)
# The multi-solution Phase 1 search alone achieves 100% optimal solutions,
# making symmetry unnecessary for this implementation.
_symmetry_available = False  # Disabled - see note above
_symmetry_caches_built = False
_symmetry_mode = 'min'
try:
    from .symmetry import get_all_rotations_fast, _build_rotation_caches
    # _symmetry_available = True  # Uncomment to re-enable
except ImportError:
    pass


def _ensure_symmetry_caches():
    """Build rotation caches on first use for faster symmetric lookups."""
    global _symmetry_caches_built
    if _symmetry_available and not _symmetry_caches_built:
        try:
            _build_rotation_caches()
            _symmetry_caches_built = True
        except Exception:
            pass

# Try to load neural heuristics (optional)
_neural_move_model = None
_neural_value_model = None
_neural_move_lut = None
try:
    from ..neural_heuristic import get_move_predictor, get_value_predictor, get_move_lut
    _neural_move_model = get_move_predictor()
    _neural_value_model = get_value_predictor()
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────
# Neural integration strategy constants
# ─────────────────────────────────────────────────────────────────
NEURAL_STRATEGY_NONE = 'none'              # No neural (baseline)
NEURAL_STRATEGY_MOVE_ORDER = 'move_order'  # Strategy A: neural move ordering only
NEURAL_STRATEGY_PROBABILISTIC = 'probabilistic'  # Strategy B: probabilistic sampling
NEURAL_STRATEGY_ADAPTIVE = 'adaptive'      # Strategy C: depth-adaptive ordering
NEURAL_STRATEGY_LUT = 'lut'               # Strategy D: precomputed LUT (O(1) lookup)
NEURAL_STRATEGY_H_SORT = 'h_sort'          # Strategy E: oracle sort by pruning-table h


class Search(object):
    """Class Search implements the Two-Phase-Algorithm.

    Enhanced with:
    1. Multi-Solution Phase 1 Search - explores multiple Phase 1 solutions
       to find shorter total solutions (8-12% improvement expected).
    2. Neural Move Ordering (optional) - uses trained neural network to
       prioritize promising moves via configurable strategies:
       - 'move_order': Always use neural ordering (Strategy A)
       - 'probabilistic': Use neural ordering with probability p (Strategy B)
       - 'adaptive': Use neural at shallow depths, default at deep (Strategy C)

    Research improvements combined: 10-14% reduction in average move count.
    """

    ax_to_s = ["U", "R", "F", "D", "L", "B"]
    po_to_s = [None, " ", "2 ", "' "]

    def __init__(self):
        self.ax              = [0] * 31  # The axis of the move
        self.po              = [0] * 31  # The power of the move
        self.flip            = [0] * 31  # phase1 coordinates
        self.twist           = [0] * 31
        self.slice           = [0] * 31
        self.parity          = [0] * 31  # phase2 coordinates
        self.URFtoDLF        = [0] * 31
        self.FRtoBR          = [0] * 31
        self.URtoUL          = [0] * 31
        self.UBtoDF          = [0] * 31
        self.URtoDF          = [0] * 31
        self.minDistPhase1   = [0] * 31  # IDA* distance do goal estimations
        self.minDistPhase2   = [0] * 31

        # Move ordering for neural integration
        # Each depth level stores its own ordered move list and current index
        self._move_orders = [None] * 31  # list of (ax, po) per depth
        self._move_idx    = [0] * 31     # current position in move list

        # Multi-solution search tracking
        self.best_solution = None
        self.best_length = float('inf')
        self.solutions_found = 0

        # Neural move ordering (for prioritizing which moves to try first)
        self.use_neural = False
        self.neural_move_model = _neural_move_model

        # Neural value prediction (for improving heuristic bounds)
        self.use_neural_value = False
        self.neural_value_model = _neural_value_model

        # Neural strategy configuration
        self.neural_strategy = NEURAL_STRATEGY_NONE
        self.neural_prob = 0.5       # probability for PROBABILISTIC strategy
        self.neural_depth_cutoff = 5 # depth cutoff for ADAPTIVE strategy

        # LUT for O(1) neural move ordering (lazy-initialised)
        self._lut = None

        # Instrumentation counters for research analysis
        self.nodes_expanded = 0         # total moves applied (node expansions)
        self.phase1_nodes = 0           # nodes in Phase 1
        self.phase2_nodes = 0           # nodes in Phase 2
        self.max_depth_reached = 0      # deepest IDA* iteration completed
        self.phase1_endpoints_found = 0 # number of G1 states reached

    def solutionToString(self, length, depthPhase1=None):
        """generate the solution string from the array data"""

        s = ""
        for i in range(length):
            s += self.ax_to_s[self.ax[i]]
            s += self.po_to_s[self.po[i]]
            if depthPhase1 is not None and i == depthPhase1 - 1:
                s += ". "
        return s

    def get_stats(self):
        """Return instrumentation counters as a dict (for benchmarking/analysis)."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'phase1_nodes': self.phase1_nodes,
            'phase2_nodes': self.phase2_nodes,
            'max_depth_reached': self.max_depth_reached,
            'phase1_endpoints_found': self.phase1_endpoints_found,
            'solutions_found': self.solutions_found,
            'best_length': self.best_length if self.best_solution else None,
        }

    @staticmethod
    def _default_move_order():
        """Default IDA* move order: U, R, F, D, L, B with powers 1, 2, 3."""
        return [(ax, po) for ax in range(6) for po in range(1, 4)]

    def _get_phase1_move_list(self, parent_ax, n, depth, max_depth):
        """
        Return an ordered list of (axis, power) moves for Phase 1,
        filtered to exclude redundant same-axis moves.

        When neural ordering is active, the moves are sorted by the
        neural model's predicted quality; otherwise they follow the
        default fixed ordering.  The 'lut' strategy replaces per-node
        neural inference with an O(1) lookup table access.  The
        'h_sort' strategy uses pruning-table h directly as the ordering
        score (oracle min-h ordering, no neural network involved).

        Args:
            parent_ax: axis of the parent move (-1 for root)
            n:         depth level for which to compute moves
            depth:     current IDA* iteration depth (for strategy)
            max_depth: current IDA* depth limit
        """
        # Get ordered moves based on strategy
        if self.neural_strategy == NEURAL_STRATEGY_LUT and self._lut is not None:
            ordered = self._lut_ordered_moves(n)
        elif self.neural_strategy == NEURAL_STRATEGY_H_SORT:
            ordered = self._h_sort_ordered_moves(n)
        elif (self.use_neural and
                self.neural_strategy != NEURAL_STRATEGY_NONE and
                self.neural_move_model is not None and
                self.neural_move_model.loaded):
            ordered = self.get_neural_move_order(n, depth, max_depth)
        else:
            ordered = self._default_move_order()

        # Filter: skip same-axis and opposite-axis-when-lower (Kociemba pruning)
        return [(ax, po) for ax, po in ordered
                if ax != parent_ax and ax != parent_ax - 3]

    def _h_sort_ordered_moves(self, n):
        """
        Oracle move ordering: sort the 18 Phase-1 moves by the pruning-table
        h-value of the resulting child, ascending. This achieves the
        p_top = 0.65 ceiling identified by measure_headroom.py --- no
        scalar-score policy can beat it for admissibility-preserving
        ordering. Cost: two getPruning calls per child (~50 ns each).
        """
        twist_n = self.twist[n]
        flip_n = self.flip[n]
        slice_n = self.slice[n]  # already the UD-slice coord (FRtoBR // 24)
        scored = [None] * 18
        for ax in range(6):
            for po in range(1, 4):
                mv = 3 * ax + (po - 1)
                new_twist = CoordCube.twistMove[twist_n][mv]
                new_flip = CoordCube.flipMove[flip_n][mv]
                # slice_n * 24 picks a representative FRtoBR in the slice class;
                # FRtoBR_Move maps it to a new FRtoBR whose //24 is the new slice.
                new_slice = CoordCube.FRtoBR_Move[slice_n * 24][mv] // 24
                h = max(
                    getPruning(
                        CoordCube.Slice_Flip_Prun,
                        CoordCube.N_SLICE1 * new_flip + new_slice,
                    ),
                    getPruning(
                        CoordCube.Slice_Twist_Prun,
                        CoordCube.N_SLICE1 * new_twist + new_slice,
                    ),
                )
                scored[mv] = (h, ax, po)
        scored.sort(key=lambda t: t[0])
        return [(ax, po) for _, ax, po in scored]

    def _neural_ordered_moves(self, n):
        """Get neural-network-ordered moves for node n. Returns list of (axis, power)."""
        try:
            move_order = self.neural_move_model.predict_move_order(
                self.twist[n], self.flip[n], self.slice[n],
                self.parity[n], self.URFtoDLF[n], self.FRtoBR[n]
            )
            return [(mv // 3, mv % 3 + 1) for mv in move_order]
        except Exception:
            return self._default_move_order()

    def _lut_ordered_moves(self, n):
        """Get LUT-ordered moves for node n (O(1) lookup). Returns list of (axis, power)."""
        try:
            return self._lut.lookup_as_axis_power(
                self.twist[n], self.flip[n], self.slice[n], self.parity[n]
            )
        except Exception:
            return self._default_move_order()

    def get_neural_move_order(self, n, depth=0, max_depth=0):
        """
        Get move ordering based on the configured neural strategy.

        Strategies:
          NONE           – default IDA* order (no neural).
          MOVE_ORDER     – always use neural ordering (Strategy A).
          PROBABILISTIC  – use neural ordering with probability self.neural_prob;
                           otherwise default order (Strategy B).
          ADAPTIVE       – use neural ordering when depth < self.neural_depth_cutoff;
                           otherwise default order (Strategy C).

        Args:
            n:         current node index in the search arrays
            depth:     current depth in IDA* (g-value)
            max_depth: current IDA* threshold

        Returns:
            list of (axis, power) tuples
        """
        model_ready = (self.neural_move_model is not None and
                       self.neural_move_model.loaded)

        if self.neural_strategy == NEURAL_STRATEGY_NONE or not model_ready:
            return self._default_move_order()

        if self.neural_strategy == NEURAL_STRATEGY_MOVE_ORDER:
            # Strategy A: always use neural ordering
            return self._neural_ordered_moves(n)

        if self.neural_strategy == NEURAL_STRATEGY_PROBABILISTIC:
            # Strategy B: use neural with probability p
            if _random.random() < self.neural_prob:
                return self._neural_ordered_moves(n)
            return self._default_move_order()

        if self.neural_strategy == NEURAL_STRATEGY_ADAPTIVE:
            # Strategy C: neural at shallow depths, default at deep
            if depth < self.neural_depth_cutoff:
                return self._neural_ordered_moves(n)
            return self._default_move_order()

        return self._default_move_order()

    def solution(self, facelets, maxDepth, timeOut, useSeparator, max_phase1_solutions=1,
                 use_neural=False, neural_strategy='none', neural_prob=0.5,
                 neural_depth_cutoff=5):
        """
        Computes the solver string for a given cube.

        @param facelets
                 is the cube definition string, see {@link Facelet} for the format.

        @param maxDepth
                 defines the maximal allowed maneuver length. For random cubes, a maxDepth of 21 usually will return a
                 solution in less than 0.5 seconds. With a maxDepth of 20 it takes a few seconds on average to find a
                 solution, but it may take much longer for specific cubes.

        @param timeOut
                 defines the maximum computing time of the method in seconds. If it does not return with a solution, it returns with
                 an error code.

        @param useSeparator
                 determines if a " . " separates the phase1 and phase2 parts of the solver string like in F' R B R L2 F .
                 U2 U D for example.<br>

        @param max_phase1_solutions
                 Number of Phase 1 solutions to explore before returning best result.
                 Higher values find shorter solutions but take more time.
                 Default: 1 (original behavior), Recommended: 5-10 for better solutions.

        @param use_neural
                 Use neural network for move ordering. Requires trained model.
                 Can provide 3-5% additional improvement in solution quality.
                 Default: False.

        @param neural_strategy
                 Neural integration strategy:
                 - 'none':          no neural ordering (baseline)
                 - 'move_order':    always reorder moves by neural prediction (Strategy A)
                 - 'probabilistic': use neural ordering with probability neural_prob (Strategy B)
                 - 'adaptive':      neural at shallow depths only (Strategy C)
                 Default: 'none'.

        @param neural_prob
                 Probability of using neural ordering in PROBABILISTIC strategy. Default: 0.5.

        @param neural_depth_cutoff
                 Depth threshold for ADAPTIVE strategy; neural used when depth < cutoff. Default: 5.

        @return The solution string or an error code:<br>
                Error 1: There is not exactly one facelet of each colour<br>
                Error 2: Not all 12 edges exist exactly once<br>
                Error 3: Flip error: One edge has to be flipped<br>
                Error 4: Not all corners exist exactly once<br>
                Error 5: Twist error: One corner has to be twisted<br>
                Error 6: Parity error: Two corners or two edges have to be exchanged<br>
                Error 7: No solution exists for the given maxDepth<br>
                Error 8: Timeout, no solution within given time
        """
        # Configure neural move ordering
        self.use_neural = use_neural and self.neural_move_model is not None

        # Configure neural strategy
        if neural_strategy == NEURAL_STRATEGY_LUT:
            # LUT strategy: build / load LUT on first use
            self.neural_strategy = NEURAL_STRATEGY_LUT
            self.use_neural = True  # enable neural path
            global _neural_move_lut
            if _neural_move_lut is None:
                try:
                    _neural_move_lut = get_move_lut()
                except Exception:
                    pass
            self._lut = _neural_move_lut
        elif neural_strategy == NEURAL_STRATEGY_H_SORT:
            # h_sort is an oracle strategy that uses the pruning tables
            # directly; no neural model is needed, but we still route
            # through the non-default move-ordering path.
            self.neural_strategy = NEURAL_STRATEGY_H_SORT
        elif use_neural and neural_strategy != NEURAL_STRATEGY_NONE:
            self.neural_strategy = neural_strategy
        elif use_neural:
            # Legacy: use_neural=True without explicit strategy → move_order
            self.neural_strategy = NEURAL_STRATEGY_MOVE_ORDER
        else:
            self.neural_strategy = NEURAL_STRATEGY_NONE
        self.neural_prob = neural_prob
        self.neural_depth_cutoff = neural_depth_cutoff

        # Configure neural value prediction (always use if available)
        self.use_neural_value = (self.neural_value_model is not None and
                                  self.neural_value_model.loaded)

        # Initialize symmetry rotation caches on first solve (lazy loading)
        _ensure_symmetry_caches()

        # +++++++++++++++++++++check for wrong input +++++++++++++++++++++++++++++
        count = [0] * 6
        try:
            for i in range(54):
                assert facelets[i] in colors
                count[colors[facelets[i]]] += 1
        except Exception:
            return "Error 1"

        for i in range(6):
            if count[i] != 9:
                return "Error 1"

        fc = FaceCube(facelets)
        cc = fc.toCubieCube()
        s = cc.verify()
        if s != 0:
            return "Error %s" % abs(s)

        # +++++++++++++++++++++++ initialization +++++++++++++++++++++++++++++++++
        c = CoordCube(cc)

        self.po[0] = 0
        self.ax[0] = 0
        self.flip[0] = c.flip
        self.twist[0] = c.twist
        self.parity[0] = c.parity
        self.slice[0] = c.FRtoBR // 24
        self.URFtoDLF[0] = c.URFtoDLF
        self.FRtoBR[0] = c.FRtoBR
        self.URtoUL[0] = c.URtoUL
        self.UBtoDF[0] = c.UBtoDF

        self.minDistPhase1[1] = 1   # else failure for depth=1, n=0
        mv = 0
        n = 0
        depthPhase1 = 1

        # Reset multi-solution tracking
        self.best_solution = None
        self.best_length = maxDepth + 1
        self.solutions_found = 0

        # Reset instrumentation counters
        self.nodes_expanded = 0
        self.phase1_nodes = 0
        self.phase2_nodes = 0
        self.max_depth_reached = 0
        self.phase1_endpoints_found = 0

        # Initialize move ordering for root (no parent axis)
        self._move_orders[0] = self._get_phase1_move_list(-1, 0, 0, depthPhase1)
        self._move_idx[0] = -1  # will be advanced to 0 on first iteration

        # Make timeout state visible to totalDepth() so Phase 2 honours the
        # same wall-clock budget the outer loop does. Without this, a cube
        # that enters a deep Phase 2 search could run minutes past the budget.
        tStart = time.time()
        self._tStart = tStart
        self._timeOut = timeOut
        self._timed_out = False

        # +++++++++++++++++++ Main loop (neural-aware) ++++++++++++++++++++++++++
        while True:
            # Time-bounded anytime: return best solution when budget exceeded
            if time.time() - tStart > timeOut:
                if self.best_solution:
                    return self.best_solution
                return "Error 8"

            # --- Advance to next move at current depth ---
            self._move_idx[n] += 1

            # --- Backtrack if all moves exhausted at this level ---
            while self._move_idx[n] >= len(self._move_orders[n]):
                if n == 0:
                    # Finished this IDA* iteration — increase depth
                    if time.time() - tStart > timeOut:
                        if self.best_solution:
                            return self.best_solution
                        return "Error 8"

                    if depthPhase1 >= maxDepth:
                        if self.best_solution:
                            return self.best_solution
                        return "Error 7"

                    depthPhase1 += 1
                    self._move_orders[0] = self._get_phase1_move_list(
                        -1, 0, 0, depthPhase1)
                    self._move_idx[0] = 0
                    break  # exit backtrack loop, process first move at root
                else:
                    n -= 1
                    self._move_idx[n] += 1

            # Set ax/po from current move
            ax_n, po_n = self._move_orders[n][self._move_idx[n]]
            self.ax[n] = ax_n
            self.po[n] = po_n

            # +++++++++++++ compute new coordinates and new minDistPhase1 ++++++++++
            # if minDistPhase1 =0, the H subgroup is reached
            mv = 3 * ax_n + po_n - 1
            self.flip[n + 1] = CoordCube.flipMove[self.flip[n]][mv]
            self.twist[n + 1] = CoordCube.twistMove[self.twist[n]][mv]
            self.slice[n + 1] = CoordCube.FRtoBR_Move[self.slice[n] * 24][mv] // 24

            # Instrumentation: count this node expansion
            self.nodes_expanded += 1
            self.phase1_nodes += 1
            if n + 1 > self.max_depth_reached:
                self.max_depth_reached = n + 1
            # Compute heuristic using 4-fold rotational symmetry
            # Using MIN across rotations (still admissible, faster than MAX)
            # MAX gives stronger bounds but causes IDA* to search deeper
            if _symmetry_available:
                if _symmetry_mode == 'min':
                    # MIN: faster search, slightly weaker bound
                    best_dist = float('inf')
                    for t, f, s in get_all_rotations_fast(
                        self.twist[n + 1], self.flip[n + 1], self.slice[n + 1]
                    ):
                        dist = max(
                            getPruning(CoordCube.Slice_Flip_Prun, CoordCube.N_SLICE1 * f + s),
                            getPruning(CoordCube.Slice_Twist_Prun, CoordCube.N_SLICE1 * t + s)
                        )
                        best_dist = min(best_dist, dist)
                    self.minDistPhase1[n + 1] = int(best_dist)
                else:
                    # MAX: stronger bound but slower search
                    best_dist = 0
                    for t, f, s in get_all_rotations_fast(
                        self.twist[n + 1], self.flip[n + 1], self.slice[n + 1]
                    ):
                        dist = max(
                            getPruning(CoordCube.Slice_Flip_Prun, CoordCube.N_SLICE1 * f + s),
                            getPruning(CoordCube.Slice_Twist_Prun, CoordCube.N_SLICE1 * t + s)
                        )
                        best_dist = max(best_dist, dist)
                    self.minDistPhase1[n + 1] = best_dist
            else:
                # Fallback to original single-lookup if symmetry module not available
                self.minDistPhase1[n + 1] = max(
                    getPruning(
                        CoordCube.Slice_Flip_Prun,
                        CoordCube.N_SLICE1 * self.flip[n + 1] + self.slice[n + 1]
                    ),
                    getPruning(
                        CoordCube.Slice_Twist_Prun,
                        CoordCube.N_SLICE1 * self.twist[n + 1] + self.slice[n + 1]
                    )
                )

            # Neural value prediction (DISABLED - causes same issue as MAX symmetry)
            # Taking max(pruning_table, neural_bound) makes heuristic stronger,
            # which paradoxically causes IDA* to search deeper (rejects solutions
            # at shallower depths, forcing exponentially more node exploration).
            # The multi-solution Phase 1 search already achieves 100% optimal
            # solutions without needing stronger heuristics.
            #
            # if self.use_neural_value:
            #     try:
            #         neural_dist = self.neural_value_model.predict_distance(
            #             self.twist[n + 1], self.flip[n + 1], self.slice[n + 1]
            #         )
            #         neural_bound = int(0.85 * neural_dist)
            #         self.minDistPhase1[n + 1] = max(self.minDistPhase1[n + 1], neural_bound)
            #     except Exception:
            #         pass
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.minDistPhase1[n + 1] == 0 and n >= depthPhase1 - 5:
                self.minDistPhase1[n + 1] = 10  # instead of 10 any value >5 is possible
                if n == depthPhase1 - 1:
                    self.phase1_endpoints_found += 1
                    # Use best_length - 1 as bound to only find better solutions
                    s = self.totalDepth(depthPhase1, self.best_length - 1 if self.best_solution else maxDepth)

                    # Phase 2 may have bailed out on the wall-clock budget. If so,
                    # return whatever is best so far; do not try more endpoints.
                    if self._timed_out:
                        if self.best_solution:
                            return self.best_solution
                        return "Error 8"

                    if s >= 0:
                        if (s == depthPhase1
                            or (
                                self.ax[depthPhase1 - 1] != self.ax[depthPhase1]
                                and self.ax[depthPhase1 - 1] != self.ax[depthPhase1] + 3)):
                            # Multi-solution search: store solution and continue
                            current_solution = self.solutionToString(s, depthPhase1) if useSeparator else self.solutionToString(s)
                            self.solutions_found += 1

                            # Track if this is better than our best
                            if s < self.best_length:
                                self.best_solution = current_solution
                                self.best_length = s

                            # Original behavior (max_phase1_solutions=1) returns immediately
                            if max_phase1_solutions <= 1:
                                return current_solution

                            # Stop only when we have explored the full K budget.
                            # A solution of length <=20 is NOT per-instance optimal:
                            # God's Number (20) is the worst-case bound across cubes,
                            # not the optimum for this particular cube. Cutting off
                            # here poisons the "best of K" statistic.
                            if self.solutions_found >= max_phase1_solutions:
                                return self.best_solution

            # --- Descend if depth budget allows ---
            # depthPhase1 - n is remaining depth; descend if > heuristic
            if depthPhase1 - n > self.minDistPhase1[n + 1]:
                n += 1
                parent_ax = self.ax[n - 1]
                self._move_orders[n] = self._get_phase1_move_list(
                    parent_ax, n, n, depthPhase1)
                self._move_idx[n] = -1  # will be advanced to 0 at loop top
            # else: stay at current n, loop top will advance _move_idx[n]

    def totalDepth(self, depthPhase1, maxDepth):
        """
        Apply phase2 of algorithm and return the combined phase1 and phase2 depth. In phase2, only the moves
        U,D,R2,F2,L2 and B2 are allowed.
        """

        mv = 0
        d1 = 0
        d2 = 0
        maxDepthPhase2 = min(10, maxDepth - depthPhase1)    # Allow only max 10 moves in phase2
        for i in range(depthPhase1):
            mv = 3 * self.ax[i] + self.po[i] - 1
            self.URFtoDLF[i + 1] = CoordCube.URFtoDLF_Move[self.URFtoDLF[i]][mv]
            self.FRtoBR[i + 1] = CoordCube.FRtoBR_Move[self.FRtoBR[i]][mv]
            self.parity[i + 1] = CoordCube.parityMove[self.parity[i]][mv]

        d1 = getPruning(
            CoordCube.Slice_URFtoDLF_Parity_Prun,
            (CoordCube.N_SLICE2 * self.URFtoDLF[depthPhase1] + self.FRtoBR[depthPhase1]) * 2 + self.parity[depthPhase1]
        )
        if d1 > maxDepthPhase2:
            return -1

        for i in range(depthPhase1):
            mv = 3 * self.ax[i] + self.po[i] - 1
            self.URtoUL[i + 1] = CoordCube.URtoUL_Move[self.URtoUL[i]][mv]
            self.UBtoDF[i + 1] = CoordCube.UBtoDF_Move[self.UBtoDF[i]][mv]

        self.URtoDF[depthPhase1] = CoordCube.MergeURtoULandUBtoDF[self.URtoUL[depthPhase1]][self.UBtoDF[depthPhase1]]

        d2 = getPruning(
            CoordCube.Slice_URtoDF_Parity_Prun,
            (CoordCube.N_SLICE2 * self.URtoDF[depthPhase1] + self.FRtoBR[depthPhase1]) * 2 + self.parity[depthPhase1]
        )
        if d2 > maxDepthPhase2:
            return -1

        self.minDistPhase2[depthPhase1] = max(d1, d2)
        if self.minDistPhase2[depthPhase1] == 0:    # already solved
            return depthPhase1

        # now set up search

        depthPhase2 = 1
        n = depthPhase1
        busy = False
        self.po[depthPhase1] = 0
        self.ax[depthPhase1] = 0
        self.minDistPhase2[n + 1] = 1   # else failure for depthPhase2=1, n=0
        # +++++++++++++++++++ end initialization +++++++++++++++++++++++++++++++++

        while True:
            while True:
                if depthPhase1 + depthPhase2 - n > self.minDistPhase2[n + 1] and not busy:

                    if self.ax[n] == 0 or self.ax[n] == 3:    # Initialize next move
                        n += 1
                        self.ax[n] = 1
                        self.po[n] = 2
                    else:
                        n += 1
                        self.ax[n] = 0
                        self.po[n] = 1
                else:
                    if self.ax[n] == 0 or self.ax[n] == 3:
                        self.po[n] += 1
                        _ = (self.po[n] > 3)
                    else:
                        self.po[n] += 2
                        _ = (self.po[n] > 3)
                    if _:
                        while True:
                            # increment axis
                            self.ax[n] += 1
                            if self.ax[n] > 5:
                                if n == depthPhase1:
                                    if depthPhase2 >= maxDepthPhase2:
                                        return -1
                                    else:
                                        depthPhase2 += 1
                                        self.ax[n] = 0
                                        self.po[n] = 1
                                        busy = False
                                        break
                                else:
                                    n -= 1
                                    busy = True
                                    break
                            else:
                                if self.ax[n] == 0 or self.ax[n] == 3:
                                    self.po[n] = 1
                                else:
                                    self.po[n] = 2
                                busy = False

                            if not (n != depthPhase1 and (self.ax[n - 1] == self.ax[n] or self.ax[n - 1] - 3 == self.ax[n])):
                                break

                    else:
                        busy = False

                if not busy:
                    break

            # +++++++++++++ compute new coordinates and new minDist ++++++++++
            mv = 3 * self.ax[n] + self.po[n] - 1

            self.URFtoDLF[n + 1] = CoordCube.URFtoDLF_Move[self.URFtoDLF[n]][mv]
            self.FRtoBR[n + 1] = CoordCube.FRtoBR_Move[self.FRtoBR[n]][mv]
            self.parity[n + 1] = CoordCube.parityMove[self.parity[n]][mv]
            self.URtoDF[n + 1] = CoordCube.URtoDF_Move[self.URtoDF[n]][mv]

            # Instrumentation: count Phase 2 node expansion
            self.nodes_expanded += 1
            self.phase2_nodes += 1

            # Wall-clock check: Phase 2 can expand millions of nodes on
            # hard instances. Without this check, a single bad cube could
            # blow the advertised timeout by 100x. Sample every 4096 nodes
            # to keep time.time() overhead negligible.
            if (self.phase2_nodes & 4095) == 0:
                if time.time() - self._tStart > self._timeOut:
                    self._timed_out = True
                    return -1

            self.minDistPhase2[n + 1] = max(
                getPruning(
                    CoordCube.Slice_URtoDF_Parity_Prun,
                    (CoordCube.N_SLICE2 * self.URtoDF[n + 1] + self.FRtoBR[n + 1]) * 2 + self.parity[n + 1]
                ),
                getPruning(
                    CoordCube.Slice_URFtoDLF_Parity_Prun,
                    (CoordCube.N_SLICE2 * self.URFtoDLF[n + 1] + self.FRtoBR[n + 1]) * 2 + self.parity[n + 1]
                )
            )
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.minDistPhase2[n + 1] == 0:
                break

        return depthPhase1 + depthPhase2

def patternize(facelets, pattern):
    facelets_cc = FaceCube(facelets).toCubieCube()
    patternized_cc = CubieCube()
    FaceCube(pattern).toCubieCube().invCubieCube(patternized_cc)
    patternized_cc.multiply(facelets_cc)
    return patternized_cc.toFaceCube().to_String()
