"""
Standalone report (not a pytest suite): computes the binary-encoding
adjacency-block LP separating margin for every distinct grid map shape in
quantum/maps/synthetic, under two encodings:

- baseline: identity code (row-major i*N+j), staying in place NOT legal --
  this is what test_binary_encoding.test_adjacency_lp_separability_boundary
  already covers for plain grids; reproduced here per real map for
  comparison against the candidate.
- candidate: split-field code (code = next_pow2(N)*i + j, so row and column
  bits occupy disjoint fields), staying in place IS legal (self-loop added
  to adjacency) -- see BINARY_ENCODING_FINDINGS.md for why both changes
  were needed together.

Reuses _lp_margin_for_code_map from test_binary_encoding_lp_code_assignment
("my function" -- the same standalone LP solver already validated against
the known 2x3/3x3/5x5 identity-case margins) rather than re-deriving the LP.

Obstacle cells still get a code (matches Grid.build_adjacency: obstacle
cells keep an adjacency entry, just with fewer neighbors -- obstacle
avoidance is a separate, still binary-unimplemented penalty, not an
adjacency-graph restriction), so obstacle maps are included as-is.

Terrain/elevation-only map variants share the same (M, N, obstacles) as
their plain counterpart and don't affect adjacency, so they're deduplicated
down to one entry per distinct (M, N, obstacles) shape.

Sizes whose truth table (2^(2B) rows) exceeds MAX_TRUTH_TABLE_ROWS are
skipped with a reason instead of silently omitted or hung on for hours --
50x50/100x100/1000x1000 land here (2^24 / 2^28 / 2^40 rows).

The LP's feasible region is a cone (every constraint is homogeneous in the
joint (coefficients, margin) vector), so the optimal margin scales exactly
linearly with `cap` (verified empirically: cap=1/5/10/25 all give the same
margin/cap to 6 decimal places). The raw margin is therefore an artifact of
the arbitrary cap=5.0 choice; `margin/cap` is the actual cap-invariant
separation strength, and it's the number that maps onto hardware relevance
-- it's directly comparable to a solver's relative coefficient precision
(e.g. D-Wave-class annealers are commonly quoted at a few percent relative
coefficient noise, so a margin/cap ratio of a fraction of a percent is
separated on paper but likely unusable on that hardware, even though
perfectly fine for simulated annealing/exact solvers). Both are reported
here so the two questions ("does a separator exist" vs "is it strong enough
for a given solver's precision") stay distinguishable.

Usage:
    python quantum/tests/binary_margin_report.py

Writes quantum/tests/binary_margin_report.json next to this script.
"""
import glob
import json
import math
import os

import yaml

from quantum.map import Grid
from quantum.tests.test_binary_encoding_lp_code_assignment import _lp_margin_for_code_map

MAX_TRUTH_TABLE_ROWS = 2**18  # 262,144 -- comfortably fast, covers up to 10x10
LP_CAP = 5.0  # matches _lp_margin_for_code_map's default; ratio is cap-invariant regardless


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def bit_width(n):
    return 0 if n <= 1 else math.ceil(math.log2(n))


def discover_map_shapes():
    """Returns {(M, N, obstacles_tuple): representative_path}, deduplicated --
    terrain/elevation variants of the same grid+obstacles collapse to one
    entry since they don't change adjacency."""
    shapes = {}
    for path in sorted(glob.glob("quantum/maps/synthetic/**/*.yaml", recursive=True)):
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        grid = (d.get("map") or {}).get("grid")
        if not grid or grid.get("M") is None or grid.get("N") is None:
            continue
        obstacles = tuple(sorted(tuple(o) for o in (grid.get("obstacles") or [])))
        key = (grid["M"], grid["N"], obstacles)
        shapes.setdefault(key, path)
    return shapes


def with_self_loops(adjacency):
    return {cell: list(neighbors) + [cell] for cell, neighbors in adjacency.items()}


def compute_case(M, N, obstacles):
    grid = Grid(M=M, N=N, obstacles=list(obstacles))
    cells = [(i, j) for i in range(M) for j in range(N)]
    num_positions = M * N

    B_identity = bit_width(num_positions)
    B_split = bit_width(next_pow2(M)) + bit_width(next_pow2(N))

    rows_identity = 2 ** (2 * B_identity)
    rows_split = 2 ** (2 * B_split)

    result = {
        "M": M, "N": N, "num_positions": num_positions,
        "obstacles": len(obstacles),
        "B_identity": B_identity, "truth_table_rows_identity": rows_identity,
        "B_split": B_split, "truth_table_rows_split": rows_split,
    }

    if rows_identity > MAX_TRUTH_TABLE_ROWS or rows_split > MAX_TRUTH_TABLE_ROWS:
        result["computed"] = False
        result["reason"] = (
            f"truth table too large to enumerate "
            f"(identity=2^{2*B_identity}={rows_identity:,} rows, "
            f"split=2^{2*B_split}={rows_split:,} rows; "
            f"cutoff={MAX_TRUTH_TABLE_ROWS:,})"
        )
        return result

    cell_to_code_identity = {c: code for code, c in enumerate(cells)}
    stride = next_pow2(N)
    cell_to_code_split = {(i, j): stride * i + j for i in range(M) for j in range(N)}

    margin_baseline = _lp_margin_for_code_map(B_identity, cell_to_code_identity, grid.adjacency, cap=LP_CAP)
    margin_candidate = _lp_margin_for_code_map(B_split, cell_to_code_split, with_self_loops(grid.adjacency), cap=LP_CAP)

    result["computed"] = True
    result["lp_cap"] = LP_CAP
    result["margin_baseline_identity_no_stay"] = margin_baseline
    result["margin_baseline_ratio"] = margin_baseline / LP_CAP if margin_baseline is not None else None
    result["margin_candidate_split_stay_legal"] = margin_candidate
    result["margin_candidate_ratio"] = margin_candidate / LP_CAP if margin_candidate is not None else None
    result["feasible_baseline"] = bool(margin_baseline and margin_baseline > 1e-9)
    result["feasible_candidate"] = bool(margin_candidate and margin_candidate > 1e-9)
    return result


def main():
    shapes = discover_map_shapes()
    report = []
    for (M, N, obstacles), path in sorted(shapes.items(), key=lambda kv: kv[0][0] * kv[0][1]):
        case = compute_case(M, N, obstacles)
        case["example_map"] = path
        report.append(case)

        if not case["computed"]:
            print(f"{M}x{N} obs={len(obstacles)}: SKIPPED -- {case['reason']}")
            continue
        print(
            f"{M}x{N} obs={len(obstacles)}: "
            f"baseline={case['margin_baseline_identity_no_stay']:.4f} "
            f"(ratio={case['margin_baseline_ratio']*100:.2f}%, "
            f"{'OK' if case['feasible_baseline'] else 'fails'})   "
            f"candidate={case['margin_candidate_split_stay_legal']:.4f} "
            f"(ratio={case['margin_candidate_ratio']*100:.2f}%, "
            f"{'OK' if case['feasible_candidate'] else 'fails'})"
        )

    out_path = os.path.join(os.path.dirname(__file__), "binary_margin_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
