# Running the tests

## Setup

Activate your project venv (same one used for the solver — see the root
`CLAUDE.md`'s Environment Setup section) and make sure the `dev` extras are
installed so `pytest` is available:

```bash
pip install -e ".[dev]"
```

## Running

Run from the **repository root** (not `quantum/`) — the test modules import
via the full package path (`from quantum.builder... import ...`), which
needs the repo root on `sys.path`. In practice pytest's own rootdir
detection (it walks up to `pyproject.toml`) makes it work from `quantum/`
too, but repo root is the safe default:

```bash
# whole test suite
python -m pytest quantum/tests -q

# one file
python -m pytest quantum/tests/test_binary_encoding.py -q

# one test function
python -m pytest quantum/tests/test_binary_encoding.py::test_start_penalty_binary_closed_form -q

# by keyword (substring match on test names)
python -m pytest quantum/tests -k "adjacency and not graph" -q
```

## Seeing what's actually happening (not just pass/fail)

pytest captures stdout by default, so any `print()` inside a test is
swallowed unless the test fails. To see it regardless of outcome:

```bash
# -s disables output capture; -v prints each test name as it runs
python -m pytest quantum/tests/test_binary_encoding.py -s -v
```

Add temporary `print(...)` calls inside a test (or in the builder methods
themselves) and rerun with `-s` — this is the quickest way to eyeball an
actual `Q` dict, a fitted block, or an LP result without leaving pytest.

## Poking at the LP/LS fit manually outside pytest

For open-ended exploration (not a fixed assertion), it's usually faster to
just drop into a Python REPL and call the builder's private fit methods
directly rather than writing a throwaway test. Minimal recipe:

```python
from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
from quantum.builder.QUBOBuilder import GridQUBOBuilder

PENALTIES = {
    "K_hot": 9, "K_adj": 4.8, "K_start": 6.5, "K_goal": 3, "K_lock": 4,
    "K_bt": 2.3, "K_tp": 1.2, "K_goal_approx": 0.7, "K_obs": 0, "K_crash": 3,
}

grid = Grid(M=5, N=5, obstacles=[])
robot = RobotConfig("r0", start=(4, 0), goal=(0, 4), expected_duration=20)
problem = PathfindingProblem([robot], grid=grid, T=20)
builder = GridQUBOBuilder(problem, PENALTIES, var_limit=9999, encoding="binary")

N = grid.N
adjacency = grid.adjacency
def neighbor_codes(code):
    i, j = divmod(code, N)
    return [k * N + l for k, l in adjacency.get((i, j), [])]

# Non-raising LP core: (success, message, margin, block) -- see
# test_adjacency_lp_separability_boundary for why this exists instead of
# just catching the RuntimeError from _fit_binary_pairwise_block_lp.
success, message, margin, block = builder._solve_binary_pairwise_lp_margin(25, neighbor_codes)
print("margin:", margin)
print(block)

# LS fit (the actual default used in production, via _BINARY_ADJACENCY_FIT_METHOD)
ls_block = builder._fit_binary_pairwise_block_ls(25, neighbor_codes)
print(ls_block)

# Full built Q for one constraint in isolation
Q = builder.build(constraints_to_apply=["start"])
print(Q)
```

For sweeping code assignments (not just the default `code = i*N+j`), reuse
`_lp_margin_for_code_map` from `test_binary_encoding_lp_code_assignment.py`
directly — it accepts an arbitrary `{cell: code}` mapping instead of
assuming real codes are `0..num_positions-1`:

```python
from quantum.tests.test_binary_encoding_lp_code_assignment import _lp_margin_for_code_map
```

## What's in this directory

- **`test_binary_encoding.py`** — regression tests for the binary-encoding
  plumbing and the constraints actually implemented so far (bit-width,
  variable indexing, start/goal-later penalty closed forms, the
  adjacency-reward LS fit's shape/caching/directional correctness), plus
  the documented LP dead end for adjacency (`_BINARY_ADJACENCY_FIT_METHOD =
  "ls"` — LP is confirmed to find no separating margin at real grid sizes).
  Fast (~2s).

- **`test_binary_encoding_lp_code_assignment.py`** — exploratory: does
  *how* codes get assigned to positions change whether LP separability is
  achievable? Tests random relabeling and genuine extra-bit code subsets
  (not just zero-padding). Both come back negative. Slower (~14s, real LP
  solves across many random trials) because it's characterizing a design
  space, not just checking implemented behavior — run it on its own if
  you're iterating on the rest of the suite and don't want to wait:

  ```bash
  python -m pytest quantum/tests/test_binary_encoding.py -q   # fast path only
  ```
