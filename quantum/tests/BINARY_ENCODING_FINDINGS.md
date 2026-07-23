# Binary encoding: investigation and verdict

**Verdict: dead end, not pursued further.** Binary position encoding
(`B = ceil(log2(N))` vars per robot/timestep instead of one-hot's `N`) cannot
support a genuine adjacency/movement constraint without more variables than
one-hot already uses. Kept on this branch as a documented negative result,
not merged into `main`.

## What's implemented here

- `BaseQUBO.binary_var_index` / `paths.bits_to_code` / `paths.decode_position_binary`:
  the encoding's plumbing (bit convention, global variable indexing, decode).
- `GridQUBOBuilder` / `GraphQUBO` binary constraints: `start`, `goal_later`,
  `adjacency_reward`. Everything else (`one_hot`, `lock`, `backtracking`,
  `tp`, `obstacle`, `multi_robot`, `adjacency_penalty`) is a stub that raises
  `NotImplementedError` — never built.
- `BaseQUBO._fit_binary_pairwise_block_lp` / `_solve_binary_pairwise_lp_margin`:
  max-margin LP fit of a shared `(2B, 2B)` adjacency block, via
  `scipy.optimize.linprog`.
- `BaseQUBO._fit_binary_pairwise_block_ls`: least-squares fit of the same
  block against BFS-distance targets. This is the production default
  (`_BINARY_ADJACENCY_FIT_METHOD = "ls"`).
- Start-position hard-fixing: `get_logical_variables()` fixes the window-
  start position's bits, and `_prepare_window` (`base_solver.py`) folds them
  out of `self.Q` via the existing generic `reduce_qubo()`. Toggle:
  `BaseQUBO._FIX_BINARY_START`.
- `decode_path(..., encoding="binary")` (`base_solver.py`): decodes each
  `(robot, t)`'s B bits into a position; a code with no corresponding real
  cell (grid/graph size isn't a power of 2) decodes to the sentinel
  `(-code, -code)` instead of aliasing onto a real cell.
- `qubo_cli.py --encoding {one_hot,binary}` wires all of the above into the
  CLI end to end.

## The core problem: adjacency has no closed form

Unlike start/goal (which collapse to a per-bit linear bias against a known
constant code), "reward moving to an adjacent cell" needs a term that reads
the *combination* of bits meaningfully differently depending on which real
cell they encode — and the same `(2B, 2B)` coefficient block gets tiled
identically across every `(robot, t)` pair and every one of the `N` possible
"from" cells.

### LP (hard max-margin separation): dead end, proven exactly

`test_adjacency_lp_separability_boundary` (`test_binary_encoding.py`)
characterizes this precisely: the LP is always *classifiable*
(`res.success=True`), but the optimal margin is exactly 0 for every grid
size tested `>= 3x4` (including exact powers of 2 like 4x4) — meaning the
best possible quadratic form over the block is no better than the trivial
all-zero one. Only tiny cases (2x3: margin 5/6, 3x3: margin 1/3) separate.
This isn't a solver-quality issue: `linprog`'s `highs` method solves this
exactly (it's a genuinely convex, exactly-solvable question — the energy of
a quadratic form at a *fixed* binary input is linear in the form's
coefficients), so margin=0 means "provably no separator exists," not
"wasn't found." An SDP relaxation of the same fitting problem can't do
better, for the same reason: there is no non-convexity here to relax against.

The real constraint is a degrees-of-freedom mismatch: `O(B²) = O(log²N)`
free coefficients have to simultaneously satisfy the ranking constraints for
all `N` "from" cells at once (same shared/tiled block). That ratio is why
2x3/3x3 work and everything bigger doesn't.

`test_binary_encoding_lp_code_assignment.py` rules out that this is an
encoding-artifact rather than a structural fact: random relabeling of codes,
and genuinely using extra bits (a random subset of a bigger code space, not
zero-padding), both still land at margin=0 for grids >= 4x4 across many
random trials.

### LS (soft fit): weak, empirically

`_fit_binary_pairwise_block_ls` always returns *a* fit (least-squares
against BFS-distance targets, no infeasibility), and the mean energy is
correctly monotonic in BFS distance (R² ≈ 0.80–0.88 across 3x3/4x4/5x5).
But per-transition discrimination is weak: pairwise correct-ordering rate
between a real valid transition and the *confusable* near-invalid class
(one hop past a real neighbor) was measured at only 55–78% across grid
sizes — barely better than chance at 5x5. Real end-to-end anneals
(`spooky-solve ... --encoding binary --penalty-set binary`) confirm this in
practice: the path collapses to the trivial low-energy state (repeated
`(0,0)`) for most timesteps regardless of `K_adj` magnitude (tried 15, 7, 6),
because nothing else in the binary formulation enforces "you must occupy a
legal position and generally keep moving" the way one-hot's
`one_hot`/`lock`/`backtracking`/`tp` do. Example run used to confirm this:

```
spooky-solve --map quantum/maps/synthetic/3x3/no_obs3x3 --problem baseline \
  --encoding binary --penalty-set binary --solver dwave \
  --benchmark --num-runs 5 --num-reads 8
```

Run with `--no-reduction-log` (`log_reductions=False`) during this testing.
Worth noting that flag doesn't materially change any of the above:
`diag_fixed_vars()` is a no-op for binary regardless (see
`get_logical_variables`), and the only other place `log_reductions` reaches
in the binary path is the start-fix's `reduce_qubo(fixed_vars,
log_reductions=builder.log_reductions)` call in `_prepare_window` — that
flag only controls whether a *reversal* log is kept for that fix (never
used here, since the start fix is never undone mid-window), not whether the
fix itself is applied.

### Why a real fix would need more variables than one-hot, not fewer

A genuine per-cell-exact fix would need an ancilla acting as an exact-match
detector for each of the `N` real "from" cells (an AND over `B` bits,
degree `B`, quadratized down to degree 2) — at best `B + N` variables per
`(robot, t)`, always strictly more than one-hot's `N` (since `B >= 1` for
any `N >= 2`):

```
 N   B    one-hot   binary   binary + ancilla (optimistic floor)
 4   2      4          2           6
 9   4      9          4          13
16   4     16          4          20
25   5     25          5          30
36   6     36          6          42
```

`tp`/`backtracking` are structurally the same problem, compounded: they need
to recognize an exact past code across non-adjacent timesteps, the same
degree-B exact-match condition as adjacency, just spanning more than two
timesteps.

## Reproducing this

See `quantum/tests/README.md` for how to run the suite and poke at the
LP/LS fit manually. `test_binary_encoding.py` is the fast (~2s) regression
suite for the plumbing and implemented constraints;
`test_binary_encoding_lp_code_assignment.py` (~14s) is the exploratory
code-assignment sweep referenced above.

## If revisited later

- One-hot's own qubit-reduction levers (BFS window pruning, `var_limit`,
  `robot_window_limits`) are proven and already qubit-competitive; that's
  where further qubit-reduction effort should go instead.
- If binary is ever revisited, the open question isn't "which fit
  algorithm" (settled — LP is exact, SDP can't beat it, LS is the best
  soft-fit option) but whether the ancilla-based exact-detection cost can be
  brought below `N` some other way; nothing tried here does.
