"""
Exploration tests: does *how* binary codes are assigned to grid positions
change LP separability of the adjacency block, beyond what
test_binary_encoding.test_adjacency_lp_separability_boundary already
characterizes for the production identity encoding (code = i*N+j, B =
paths.bit_width(num_positions))?

Two variants are tested, both negative:

1. Random relabeling: same B, same number of real codes, just a different
   bijection cells <-> codes 0..num_positions-1. If separability were a
   labeling artifact, some permutation should beat margin=0.
2. Genuine extra-bit use: B larger than the minimum, with real positions
   placed at a random SUBSET of a bigger code space (not "N real codes
   plus always-zero padding bits", which was already ruled out earlier in
   the same investigation -- here the extra bits actually vary across the
   chosen real codes, so they're not structurally inert).

Both come back margin == 0 for every trial tried at grid sizes >= 4x4,
mirroring the negative result already locked in for the identity encoding
in test_adjacency_lp_separability_boundary. This reinforces that the
bottleneck is the shared (2B, 2B) block's O(B^2) capacity against the
O(num_positions^2) distinctions the truth table demands -- not which
specific codes or how many bits get used.

Uses a hand-rolled LP margin solver (_lp_margin_for_code_map) rather than
BaseQUBO._fit_binary_pairwise_block_lp / _binary_transition_truth_table,
because those hardcode "real codes are exactly 0..num_positions-1" -- they
have no notion of an arbitrary code subset, which is exactly what this file
needs to vary.
"""
import itertools
import random
from collections import deque

import numpy as np
import pytest
from scipy.optimize import linprog

from quantum.map import Grid
from quantum.utils.paths import bit_width


def _bits_to_code(bits):
    code = 0
    for b, bit in enumerate(bits):
        code |= (bit & 1) << b
    return code


def _lp_margin_for_code_map(B, cell_to_code, adjacency, cap=5.0):
    """
    Generalized max-margin LP fit, matching the objective/constraints of
    BaseQUBO._solve_binary_pairwise_lp_margin, but cell_to_code may map
    real cells to ANY subset of the 2**B codes -- not just the contiguous
    0..num_positions-1 the production code assumes. Codes outside
    cell_to_code's range are ghost (same "one worse than the largest real
    BFS hop-distance" target as the production truth table).

    Returns the margin (float) if the LP solved to optimality, else None.
    """
    real_codes = set(cell_to_code.values())

    dist = {}
    for src_cell, src_code in cell_to_code.items():
        dist[(src_code, src_code)] = 0
        frontier = deque([src_cell])
        seen = {src_cell}
        while frontier:
            cur = frontier.popleft()
            cur_code = cell_to_code[cur]
            for nb in adjacency.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    dist[(src_code, cell_to_code[nb])] = dist[(src_code, cur_code)] + 1
                    frontier.append(nb)
    max_real_dist = max(dist.values(), default=0)
    ghost_target = max_real_dist + 1
    neighbor_of_code = {
        cell_to_code[c]: [cell_to_code[nb] for nb in adjacency.get(c, [])]
        for c in cell_to_code
    }

    def target_for(c_from, c_to):
        if c_from not in real_codes or c_to not in real_codes:
            return ghost_target
        if c_to in neighbor_of_code[c_from]:
            return 0
        neighbors = neighbor_of_code[c_from]
        if not neighbors:
            return ghost_target
        return min(dist.get((n, c_to), ghost_target) for n in neighbors)

    n_bits = 2 * B
    pair_idx = list(itertools.combinations(range(n_bits), 2))
    n_diag = n_bits
    n_pair = len(pair_idx)
    m_pos = n_diag + n_pair

    A_ub, b_ub = [], []
    for bits in itertools.product((0, 1), repeat=n_bits):
        if not any(bits):
            continue
        c_from = _bits_to_code(bits[:B])
        c_to = _bits_to_code(bits[B:])
        target = target_for(c_from, c_to)
        diag_coeffs = list(bits)
        pair_coeffs = [bits[i] * bits[j] for i, j in pair_idx]
        if target == 0:
            A_ub.append(diag_coeffs + pair_coeffs + [0])
            b_ub.append(0)
        else:
            A_ub.append([-c for c in diag_coeffs] + [-c for c in pair_coeffs] + [target])
            b_ub.append(0)

    c_obj = [0] * (n_diag + n_pair) + [-1]
    bounds = [(-cap, cap)] * (n_diag + n_pair) + [(0, cap)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[m_pos]


def _grid_cells_and_adjacency(M, N):
    grid = Grid(M=M, N=N, obstacles=[])
    cells = [(i, j) for i in range(M) for j in range(N)]
    return cells, grid.adjacency


def test_generalized_lp_margin_matches_known_identity_case_boundary():
    """Sanity check that _lp_margin_for_code_map (a standalone
    reimplementation) agrees with the production
    _solve_binary_pairwise_lp_margin's already-confirmed values under the
    identity mapping, before trusting it for the exploration below: 5/6 for
    2x3, 1/3 for 3x3, exactly 0 for 5x5."""
    known = [(2, 3, 5 / 6), (3, 3, 1 / 3), (5, 5, 0.0)]
    for M, N, expected_margin in known:
        cells, adjacency = _grid_cells_and_adjacency(M, N)
        B = bit_width(M * N)
        cell_to_code = {cell: code for code, cell in enumerate(cells)}
        margin = _lp_margin_for_code_map(B, cell_to_code, adjacency)
        assert margin == pytest.approx(expected_margin, abs=1e-6)


def test_random_relabeling_does_not_rescue_separability():
    """Same bit-width as the production identity encoding, just a
    different bijection from cells to codes 0..num_positions-1. If
    separability were a labeling artifact, some permutation should find
    margin > 0 for these grids -- none does."""
    rng = random.Random(0)
    cases = [(4, 4, 40), (3, 4, 40), (5, 5, 20)]
    for M, N, n_trials in cases:
        cells, adjacency = _grid_cells_and_adjacency(M, N)
        num_positions = M * N
        B = bit_width(num_positions)
        best_margin = 0.0
        for _ in range(n_trials):
            perm = cells[:]
            rng.shuffle(perm)
            cell_to_code = {cell: code for code, cell in enumerate(perm)}
            margin = _lp_margin_for_code_map(B, cell_to_code, adjacency)
            assert margin is not None, f"{M}x{N}: LP was not classifiable for some permutation"
            best_margin = max(best_margin, margin)
        assert best_margin == pytest.approx(0.0, abs=1e-9), (
            f"{M}x{N}: found a permutation with margin {best_margin} > 0 -- "
            f"relabeling alone would rescue separability, contradicting the "
            f"prior finding that it doesn't"
        )


def test_extra_bits_with_nontrivial_code_subset_does_not_rescue_separability():
    """Genuinely uses the extra bits: instead of the minimum-B codes
    0..num_positions-1, picks a random num_positions-sized SUBSET of a
    larger 2**(B+extra) code space, so the extra bits actually vary across
    the chosen real codes (unlike zero-padding, where they're constant and
    structurally inert). Still margin == 0 for every trial at 4x4 and 5x5
    with up to 2 extra bits -- confirms the bottleneck is the shared
    block's capacity, not which/how-many bits get used."""
    rng = random.Random(1)
    # (M, N, extra_bits, trials) -- trial counts tuned to keep the whole
    # file's runtime in the single-digit seconds; LP cost grows fast with B
    # (2**(2B) truth-table rows), so higher-B cases get fewer trials.
    cases = [
        (4, 4, 1, 15),
        (4, 4, 2, 10),
        (5, 5, 1, 10),
        (5, 5, 2, 5),
    ]
    for M, N, extra, n_trials in cases:
        cells, adjacency = _grid_cells_and_adjacency(M, N)
        num_positions = M * N
        B = bit_width(num_positions) + extra
        num_codes = 2 ** B
        best_margin = 0.0
        for _ in range(n_trials):
            codes = rng.sample(range(num_codes), num_positions)
            shuffled_cells = cells[:]
            rng.shuffle(shuffled_cells)
            cell_to_code = dict(zip(shuffled_cells, codes))
            margin = _lp_margin_for_code_map(B, cell_to_code, adjacency)
            assert margin is not None, f"{M}x{N} B={B}: LP was not classifiable for some code subset"
            best_margin = max(best_margin, margin)
        assert best_margin == pytest.approx(0.0, abs=1e-9), (
            f"{M}x{N} B={B} (+{extra} bits, {num_codes} codes available): "
            f"found margin {best_margin} > 0 -- extra bits would rescue "
            f"separability, contradicting the prior finding that they don't"
        )
