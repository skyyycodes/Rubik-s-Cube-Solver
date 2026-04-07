"""
4-fold rotational symmetry for improved Phase 1 heuristics.

Rotates cube coordinates around Z-axis (U-D axis) by 90°, 180°, 270°.
This allows checking multiple symmetric positions to get better heuristic estimates.

Key insight: The UD-slice coordinate is INVARIANT under Z-rotation, so we only
need to transform twist and flip coordinates.
"""

# Corner permutation under 90° clockwise Z-rotation (looking from above)
# Position i gets its value from position CORNER_ROT_SRC[i]
# After rotation: F→R, R→B, B→L, L→F, U stays, D stays
# URF(0)→UBR(3), UFL(1)→URF(0), ULB(2)→UFL(1), UBR(3)→ULB(2)
# DFR(4)→DRB(7), DLF(5)→DFR(4), DBL(6)→DLF(5), DRB(7)→DBL(6)
CORNER_ROT_SRC = [1, 2, 3, 0, 5, 6, 7, 4]

# Edge permutation under 90° clockwise Z-rotation
# UR(0)→UB(3), UF(1)→UR(0), UL(2)→UF(1), UB(3)→UL(2)
# DR(4)→DB(7), DF(5)→DR(4), DL(6)→DF(5), DB(7)→DL(6)
# FR(8)→BR(11), FL(9)→FR(8), BL(10)→FL(9), BR(11)→BL(10)
EDGE_ROT_SRC = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8]

# Powers of 3 for twist encoding
POW3 = [1, 3, 9, 27, 81, 243, 729]  # 3^0 to 3^6

# Powers of 2 for flip encoding
POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # 2^0 to 2^10


def decompose_twist(twist):
    """Decompose twist coordinate into corner orientations (0-2 each)."""
    co = [0] * 8
    total = 0
    for i in range(7):
        co[i] = twist % 3
        total += co[i]
        twist //= 3
    # 8th corner determined by constraint: sum of orientations ≡ 0 (mod 3)
    co[7] = (3 - (total % 3)) % 3
    return co


def compose_twist(co):
    """Compose corner orientations back into twist coordinate."""
    twist = 0
    for i in range(6, -1, -1):  # 6 down to 0
        twist = twist * 3 + co[i]
    return twist


def decompose_flip(flip):
    """Decompose flip coordinate into edge orientations (0-1 each)."""
    eo = [0] * 12
    total = 0
    for i in range(11):
        eo[i] = flip % 2
        total += eo[i]
        flip //= 2
    # 12th edge determined by constraint: sum of orientations ≡ 0 (mod 2)
    eo[11] = total % 2
    return eo


def compose_flip(eo):
    """Compose edge orientations back into flip coordinate."""
    flip = 0
    for i in range(10, -1, -1):  # 10 down to 0
        flip = flip * 2 + eo[i]
    return flip


def rotate_twist_90(twist):
    """Rotate twist coordinate by 90° clockwise around Z-axis."""
    co = decompose_twist(twist)
    # Permute corner orientations
    new_co = [co[CORNER_ROT_SRC[i]] for i in range(8)]
    return compose_twist(new_co)


def rotate_flip_90(flip):
    """Rotate flip coordinate by 90° clockwise around Z-axis."""
    eo = decompose_flip(flip)
    # Permute edge orientations
    new_eo = [eo[EDGE_ROT_SRC[i]] for i in range(12)]
    return compose_flip(new_eo)


def get_all_rotations(twist, flip, slice_coord):
    """
    Get all 4 rotational variants of the coordinates.

    Returns list of (twist, flip, slice) tuples for 0°, 90°, 180°, 270° rotations.
    Note: slice_coord is invariant under Z-rotation, so it's the same for all.
    """
    coords = [(twist, flip, slice_coord)]
    t, f = twist, flip

    for _ in range(3):  # 90°, 180°, 270°
        t = rotate_twist_90(t)
        f = rotate_flip_90(f)
        coords.append((t, f, slice_coord))

    return coords


# Precompute rotation tables for faster lookup (optional optimization)
# This trades memory for speed by caching all possible rotations

_twist_rot90_cache = None
_flip_rot90_cache = None


def _build_rotation_caches():
    """Build rotation lookup tables for all possible coordinate values."""
    global _twist_rot90_cache, _flip_rot90_cache

    # Only build if not already built
    if _twist_rot90_cache is not None:
        return

    # Use constants directly to avoid importing CoordCube (which loads huge tables)
    N_TWIST = 2187  # 3^7 possible corner orientations
    N_FLIP = 2048   # 2^11 possible edge flips

    # Build twist rotation cache: twist_rot90_cache[twist] = rotated_twist
    _twist_rot90_cache = [0] * N_TWIST
    for twist in range(N_TWIST):
        _twist_rot90_cache[twist] = rotate_twist_90(twist)

    # Build flip rotation cache
    _flip_rot90_cache = [0] * N_FLIP
    for flip in range(N_FLIP):
        _flip_rot90_cache[flip] = rotate_flip_90(flip)


def get_all_rotations_fast(twist, flip, slice_coord):
    """
    Get all 4 rotational variants using precomputed lookup tables.

    Call _build_rotation_caches() once at startup for this to work.
    Falls back to slow method if caches not built.
    """
    if _twist_rot90_cache is None:
        return get_all_rotations(twist, flip, slice_coord)

    coords = [(twist, flip, slice_coord)]
    t, f = twist, flip

    for _ in range(3):
        t = _twist_rot90_cache[t]
        f = _flip_rot90_cache[f]
        coords.append((t, f, slice_coord))

    return coords


def check_heuristic_consistency(n_samples=1000, verbose=False):
    """
    Test if the symmetry heuristic (MAX across rotations) violates consistency.

    Consistency property: h(n) <= cost(n->n') + h(n') for all moves
    In other words: h(child) <= h(parent) + 1

    If this is violated, IDA* will explore exponentially more nodes.

    Returns: (violations_max, violations_min, total_checks)
    """
    import random
    from .coordcube import CoordCube, getPruning

    N_TWIST = 2187
    N_FLIP = 2048
    N_SLICE1 = 495

    _build_rotation_caches()

    violations_max = 0  # Violations when using MAX across rotations
    violations_min = 0  # Violations when using MIN across rotations
    total_checks = 0

    def symmetry_heuristic_max(twist, flip, slice_c):
        """Compute heuristic using MAX across rotations."""
        best_dist = 0
        for t, f, s in get_all_rotations_fast(twist, flip, slice_c):
            dist = max(
                getPruning(CoordCube.Slice_Flip_Prun, N_SLICE1 * f + s),
                getPruning(CoordCube.Slice_Twist_Prun, N_SLICE1 * t + s)
            )
            best_dist = max(best_dist, dist)
        return best_dist

    def symmetry_heuristic_min(twist, flip, slice_c):
        """Compute heuristic using MIN across rotations."""
        best_dist = float('inf')
        for t, f, s in get_all_rotations_fast(twist, flip, slice_c):
            dist = max(
                getPruning(CoordCube.Slice_Flip_Prun, N_SLICE1 * f + s),
                getPruning(CoordCube.Slice_Twist_Prun, N_SLICE1 * t + s)
            )
            best_dist = min(best_dist, dist)
        return best_dist

    for _ in range(n_samples):
        # Generate random coordinates
        twist = random.randint(0, N_TWIST - 1)
        flip = random.randint(0, N_FLIP - 1)
        slice_c = random.randint(0, N_SLICE1 - 1)

        h_parent_max = symmetry_heuristic_max(twist, flip, slice_c)
        h_parent_min = symmetry_heuristic_min(twist, flip, slice_c)

        # Apply each of the 18 moves and check consistency
        for mv in range(18):
            new_twist = CoordCube.twistMove[twist][mv]
            new_flip = CoordCube.flipMove[flip][mv]
            new_slice = CoordCube.FRtoBR_Move[slice_c * 24][mv] // 24

            h_child_max = symmetry_heuristic_max(new_twist, new_flip, new_slice)
            h_child_min = symmetry_heuristic_min(new_twist, new_flip, new_slice)

            total_checks += 1

            # Check consistency: h(child) should be <= h(parent) + 1
            if h_child_max > h_parent_max + 1:
                violations_max += 1
                if verbose:
                    print(f"MAX violation: h({twist},{flip},{slice_c})={h_parent_max} "
                          f"-> h({new_twist},{new_flip},{new_slice})={h_child_max}")

            if h_child_min > h_parent_min + 1:
                violations_min += 1
                if verbose:
                    print(f"MIN violation: h({twist},{flip},{slice_c})={h_parent_min} "
                          f"-> h({new_twist},{new_flip},{new_slice})={h_child_min}")

    return violations_max, violations_min, total_checks


if __name__ == '__main__':
    # Test the symmetry functions
    print("Testing symmetry functions...")

    # Test decompose/compose roundtrip
    for twist in [0, 1, 100, 1000, 2186]:
        co = decompose_twist(twist)
        recomposed = compose_twist(co)
        assert twist == recomposed, f"Twist roundtrip failed: {twist} -> {co} -> {recomposed}"
    print("  Twist roundtrip: OK")

    for flip in [0, 1, 100, 1000, 2047]:
        eo = decompose_flip(flip)
        recomposed = compose_flip(eo)
        assert flip == recomposed, f"Flip roundtrip failed: {flip} -> {eo} -> {recomposed}"
    print("  Flip roundtrip: OK")

    # Test that 4 rotations bring us back to original
    for twist in [0, 100, 1000]:
        t = twist
        for _ in range(4):
            t = rotate_twist_90(t)
        assert t == twist, f"4x rotation didn't return to original: {twist} -> {t}"
    print("  Twist 4x rotation: OK")

    for flip in [0, 100, 1000]:
        f = flip
        for _ in range(4):
            f = rotate_flip_90(f)
        assert f == flip, f"4x rotation didn't return to original: {flip} -> {f}"
    print("  Flip 4x rotation: OK")

    # Test get_all_rotations
    rotations = get_all_rotations(100, 200, 50)
    assert len(rotations) == 4
    assert rotations[0] == (100, 200, 50)
    assert all(r[2] == 50 for r in rotations)  # slice invariant
    print("  get_all_rotations: OK")

    print("\nAll tests passed!")
