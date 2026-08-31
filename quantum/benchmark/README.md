# Benchmarking

This module provides tools for evaluating the performance and reliability of the quantum solvers. It allows for running repeated trials of pathfinding problems to gather statistics on success rates, solution quality, and execution time.

## Components

### `BenchmarkRunner`

The core class that orchestrates the benchmarking process.

- **Multiple Runs**: Executes the solver a specified number of times (`num_runs`) on the same problem instance.
- **Validation**: Automatically validates each solution to ensure it meets all constraints (valid moves, no collisions, correct start/end).
- **Data Collection**: Aggregates results including:
  - Success/Failure status
  - Energy of the solution
  - Execution time
  - Decoded paths
- **Storage**: Saves detailed results to JSON files in `results/benchmarks`.
- **Verbosity Levels**: Control output detail and memory usage with 3 benchmark levels.

## Benchmark Levels

The `level` parameter controls the verbosity and memory footprint of benchmark results. This is particularly useful when running large-scale benchmarks (e.g., 1000+ runs) where memory consumption can become significant.

### Level 1: Summary Only (Minimal Memory)

**Use case**: Large-scale performance testing, CI/CD pipelines, quick validation

**Includes:**

- Run ID and timestamp
- Validation pass/fail status
- Total energy
- Execution time

**Excludes:**

- Robot paths
- Raw bit solutions
- Per-robot validation details

**Memory**: ~100-200 bytes per run

### Level 2: Paths Included (Default)

**Use case**: Standard benchmarking, path analysis, debugging multi-robot coordination

**Includes:**

- Everything from Level 1
- Robot paths for each run
- Per-robot validation details
- Per-window energies (for windowed solving)

**Excludes:**

- Raw bit solutions

**Memory**: ~500-2000 bytes per run (depends on path length and number of robots)

### Level 3: Full Debug (Maximum Detail)

**Use case**: Deep debugging, QUBO analysis, solution verification

**Includes:**

- Everything from Level 2
- Raw bit solution (complete QUBO variable assignment)

**Excludes:**

- Nothing (full detail)

**Memory**: ~2000-10000 bytes per run (depends on QUBO size)

## Usage

The benchmark runner is typically invoked within a script like `qubo.py`.

### Basic Usage (Default Level 2)

```python
from quantum.benchmark import BenchmarkRunner

# Initialize benchmark with a builder and a solver
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=100,
    output_dir="results/benchmarks"
)

# Execute the benchmark
runner.run_build()
```

### Memory-Optimized Benchmarking (Level 1)

```python
# For large-scale benchmarks (1000+ runs)
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=1000,
    level=1  # Summary only - minimal memory
)
runner.run_build()
```

### Full Debug Mode (Level 3)

```python
# For debugging QUBO formulations
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=10,
    level=3  # Include raw bit solutions
)
runner.run_build()
```

## Output Format

Benchmark results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "problem": { ... },
    "solver": { ... },
    "penalty_set": { ... },
    "benchmark_level": 2,
    "num_runs": 100,
    "timestamp": "2026-01-26T20:00:00"
  },
  "runs": [
    {
      "run_id": 1,
      "timestamp": "2026-01-26T20:00:01",
      "valid": true,
      "energy": 123.45,
      "execution_time_sec": 2.5,
    
      // Level 2+ only:
      "robot_paths": {
        "Robot1": [[0, 0], [0, 1], ...],
        "Robot2": [[5, 5], [5, 4], ...]
      },
      "validation_details": { ... },
      "window_energies": [45.2, 38.1, 40.15],
    
      // Level 3 only:
      "raw_solution": [0, 1, 0, 1, 1, 0, ...]
    }
  ]
}
```

## Validation Logic

The benchmarking tool includes robust validation logic (`is_solution_valid`) that supports both Grid and Graph problem notations. It checks:

- **Continuity**: Robots must move between connected nodes/cells.
- **Obstacles**: Robots must not traverse blocked areas.
- **One-Hot**: Robots must be in exactly one place at one time.
- **Multi-Robot**: Robots must not collide with each other.
- **Start/Goal**: Robots must start and end at correct positions.
- **Early Stopping**: Supports robots reaching goals before time horizon.

## Performance Considerations

### Memory Usage Estimates

For a typical multi-robot problem with 3 robots and 20 timesteps:

| Level | Memory per Run | 100 Runs | 1000 Runs |
| ----- | -------------- | -------- | --------- |
| 1     | ~150 bytes     | ~15 KB   | ~150 KB   |
| 2     | ~1.5 KB        | ~150 KB  | ~1.5 MB   |
| 3     | ~5 KB          | ~500 KB  | ~5 MB     |

**Recommendation**: Use Level 1 for runs > 500, Level 2 for standard benchmarks, Level 3 only for debugging specific issues.

---

# Sweeps

`BenchmarkRunner` (above) benchmarks **one** problem instance against **one**
solver. A **sweep** runs the full matrix of
`instance × problem × solver × ablation`, driving `BenchmarkRunner` unchanged
for each cell, and produces a reproducibility manifest plus an index that the
analysis layer reads. Everything below is run from the **repo root**.

| File                                 | Role                                                                                                                                                  |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_sweep.py`                     | `spooky-sweep` CLI entry point.                                                                                                                     |
| `sweep_runner.py`                  | `SweepRunner` — validates the config up front, executes the matrix, checkpoints after every cell, supports `--resume`.                           |
| `manifest.py`                      | Builds`manifest.json` — git state, package versions, hardware, seed, what was skipped/failed.                                                      |
| `analysis/`                        | Post-processing: aggregate JSON → CSV tables, plots, LaTeX, HF publish. See[`analysis/README.md`](analysis/README.md) for the per-column reference. |
| `sweep_configs/*.yaml` (repo root) | The sweep definitions. Each has a header comment explaining its scope and caveats — read it before trusting output.                                  |

## Sweep config anatomy

```yaml
sweep:
  id: null                 # auto: sweep_<timestamp>_<hex8>
  seed: 42                  # seeds numpy global RNG once; auto-injected into dwave entries
  enable_hardware: false
  output_root: results/sweeps   # relative paths resolve under <repo>/

instances:
  - map: quantum/maps/synthetic/5x5/obs5x5_hard   # repo-root-relative, no extension
    problems: [baseline, two_robots]              # named problems in the sibling <map>.yaml
    builder: grid                                 # grid | graph

solvers:
  - name: sa_neal            # free-text label; the unit the analysis groups/pairs on
    backend: dwave           # must be registered in SolverFactory (dwave|pennylane|ilp|cbs)
    params: { normalize_scale: 4, num_reads: 4 }
    penalty_set: crash       # name from quantum/config/config.yaml; omit for ilp/cbs
    num_runs: 10             # 1 for deterministic backends, N for stochastic
    ablation: { preprocess: [true, false] }   # each value becomes its own matrix cell
  - name: qpu_run
    backend: dwave
    hardware: true           # gated: needs --enable-hardware AND --confirm-hardware-quota

execution:
  preprocess_default: true
  fail_fast: false           # true = abort the whole sweep on the first cell that errors
  run_timeout_sec: 600       # per-cell wall-clock cap (best-effort SIGALRM)
```

`preprocess` values may be `true`/`false` or preprocess-mode strings; they are
normalised so `index.json` records one vocabulary.

## Running a sweep

```bash
# validate config + print the execution plan, run nothing
spooky-sweep --config sweep_configs/classical_test.yaml --dry-run

# run it
spooky-sweep --config sweep_configs/classical_test.yaml

# restrict the matrix
spooky-sweep -c sweep_configs/classical_test.yaml \
    --only-solvers sa_neal,ilp_highs \
    --only-instances quantum/maps/synthetic/5x5/obs5x5_hard

# resume: bare flag auto-finds the most recent incomplete sweep whose stored
# config matches this file exactly; or pass an explicit sweep_id
spooky-sweep -c sweep_configs/classical_test.yaml --resume
spooky-sweep -c sweep_configs/classical_test.yaml --resume sweep_20260807_120000_ab12cd34

# hardware-gated entries: two independent flags on purpose, so a stray or
# copy-pasted command cannot spend QPU quota by accident
spooky-sweep -c sweep_configs/full_comparison.yaml \
    --enable-hardware --confirm-hardware-quota yes-spend-quota
```

Resume granularity is the cell: any
`(instance, problem, solver, preprocess)` that already has a completed
`benchmark_*.json` under the sweep's directory is skipped; anything that
failed or never ran is retried. The check is against the directory on disk
(deterministic cell-directory naming), not a loaded index, so it self-heals a
lost/corrupt `index.json`.

## Sweep output layout

```
results/sweeps/<sweep_id>/
├── manifest.json     provenance + reproducibility record (see below)
├── index.json        one entry per matrix cell → its benchmark_*.json path
├── <inst>__<problem>__<solver>__preprocess_<mode>/
│   └── benchmark_YYYYMMDD_HHMMSS.json     BenchmarkRunner output, level 2
├── ...
└── analysis/         created by run_aggregate / run_report
    ├── runs_long.csv                one row per individual solver run
    ├── summary_by_solver.csv        success rate + variable-reduction means
    ├── statistical_tests.csv        paired Wilcoxon tests (see next section)
    ├── energy_diagnostics.csv       per-config energy-landscape health
    ├── failure_causes.csv           invalid runs split by cause
    ├── robot_statistics_long.csv    one row per (run, robot); needs level ≥ 2
    ├── benchmark_table.tex          paper-style table snippet (run_report)
    └── plots/*.html, *.png          Plotly figures (run_plots / run_report)
```

`SPOOKY_RESULTS_DIR` relocates the whole `results/` tree; a sweep config's
`output_root` overrides just `results/sweeps` for that sweep.

## `manifest.json` — reproducibility record

Answers "what exactly produced these numbers, and can it be reproduced"
without digging through chat history. Written partially at sweep start (so a
crash still leaves a record of intent) and checkpointed after every cell.

| Field             | Meaning                                                                                                                                                                                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `git`           | `commit`, `branch`, `dirty` (uncommitted changes present).                                                                                                                                                                                                                    |
| `packages`      | Versions of the solver-relevant deps (pennylane, dimod, dwave-neal, pyomo, highspy, numpy, scipy, …).                                                                                                                                                                              |
| `hardware`      | CPU count, platform, Python version, GPU (from`nvidia-smi`).                                                                                                                                                                                                                      |
| `global_seed`   | Surfaced at top level as well as inside`sweep_config`. Not a full determinism guarantee (PennyLane has no seed hook today).                                                                                                                                                      |
| `sweep_config`  | Deep copy of the YAML as loaded.`--resume` matches on this exactly.                                                                                                                                                                                                               |
| `skipped`       | Hardware-gated entries the config wanted but that ran without`--enable-hardware`. To demonstrate the real-hardware run was *deliberately* skipped, not silently missing.                                                                                                  |
| `failures`      | Per-cell error + traceback for anything that raised (unless`fail_fast`, which re-raises).                                                                                                                                                                                         |
| `resume_events` | One fresh environment snapshot per`--resume`. The original `start_time`/`git`/`packages` are preserved.                                                                                                                                                                     |
| `end_time`      | **`null` until the run loop reaches its natural end.** A manifest read mid-sweep or after an interruption keeps `end_time: null`; only `last_checkpoint` advances during the run. This is how the analysis and the auto-resume scan tell "finished" from "interrupted". |

---

# The paired statistical test

`analysis/aggregate.py::run_statistical_tests` → **`statistical_tests.csv`**.
One row per `(solver_a, solver_b, metric)`. Default pairs are
`(candidate, baseline)` with `ilp`/`cbs` as baselines; direction is preserved
— every directional column is `solver_a` **relative to** `solver_b`.

## Why Wilcoxon signed-rank

The question is "does solver A beat solver B on this metric, across the
benchmark instances". Three properties of the data decide the test:

1. **Paired.** Both solvers run the *same* instances, and instances differ in
   difficulty by orders of magnitude. Pairing by instance and testing the
   per-instance differences removes that between-instance variance — an
   unpaired test would drown the solver effect in it.
2. **Non-parametric.** Runtimes across instances are heavy-right-tailed, path
   efficiency is bounded in `[0, 1]`, and the sample size is the number of
   instances (often 6–20). None of that supports the paired *t*-test's
   normal-differences assumption, and *n* is too small for the CLT to rescue
   it.
3. **Uses magnitude, not just sign.** The signed-rank test ranks the
   *absolute* paired differences and checks whether the positive and negative
   ones are symmetric about zero under H₀. That is strictly more powerful
   than the sign test (which discards magnitude) while assuming far less than
   a *t*-test.

So Wilcoxon signed-rank is the paired, distribution-free middle ground, and
it is the standard choice for "solver A vs solver B over a benchmark suite".

## Pairing, transforms, and the minimum

- **Pairs by instance, not by run.** For each solver, its per-instance *mean*
  of the metric over that solver's valid runs; then the paired per-instance
  values are tested. Raw per-run values from a stochastic solver on one
  instance are not independent draws suitable for a cross-solver paired test.
- **`transform="log"` for time metrics** (`execution_time_sec`,
  `build_time_sec`). Wilcoxon ranks *absolute* differences, so on a sweep
  spanning 0.1 s to 600 s the ranking would be decided entirely by the
  biggest instances — a 2× speedup on a small instance would rank below a 5%
  wobble on a large one. Testing `log(a) − log(b)` puts every instance on
  equal footing and makes the symmetry assumption plausible for a
  multiplicative quantity. `median_ratio = exp(median_diff)` reads as
  "solver_a takes N× as long as solver_b".
- **`_MIN_PAIRS_FOR_WILCOXON = 6`.** Below ~6 paired instances the two-sided
  test cannot reach p < 0.05 no matter how one-sided the data (with n = 5 the
  smallest attainable p is 1/16 ≈ 0.0625). Rows with fewer pairs get
  `note = insufficient_data` and no p-value.
- **`n_effective` vs `n_pairs`.** scipy's default `zero_method="wilcox"`
  discards paired differences of exactly zero, so the test's real *n* is
  `n_effective`. Both are reported and the minimum-pairs guard is applied to
  both. An instance where the two solvers tie contributes nothing.

## Effect size and significance

- **`rank_biserial`** — matched-pairs rank-biserial correlation,
  `(W⁺ − W⁻) / (W⁺ + W⁻)`, in `[−1, 1]`. This is the effect size that goes
  with Wilcoxon the way Cohen's *d* goes with the *t*-test; the p-value alone
  only says "some difference exists". `favors` resolves it to a solver name
  using each metric's `higher_is_better`.
- **`p_value`** is raw. **`p_value_bh`** is Benjamini–Hochberg FDR-adjusted
  across *every* row in the same `run_statistical_tests` call that produced a
  real p-value. **Use `p_value_bh` for any "is this significant" claim** — a
  sweep runs a whole family of tests (every pair × every metric) and the raw
  p-values are not corrected for that.

## Read these before believing a row

A paired test conditions the population on *both* solvers having produced a
usable value, so the set of instances behind a row silently shifts from pair
to pair. A solver that only succeeds on easy instances is compared only on
easy instances and looks both fast and accurate. That bias is intrinsic to
paired testing and cannot be removed here — it is made visible instead:

| Column                  | Meaning                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `n_dropped_unshared`  | Only one of the two solvers attempted the instance at all.                                                                                |
| `n_dropped_invalid`   | Both attempted; at least one had no valid run.                                                                                            |
| `n_dropped_no_metric` | Both had valid runs; the metric wasn't recorded (e.g. path efficiency needs`BenchmarkRunner` level ≥ 2).                               |
| `min_valid_runs`      | Fewest valid runs behind any single paired cell.`1` means some instance is represented by one lucky run out of many attempts.           |
| `n_censored`          | Paired instances where either solver hit a time/node limit. Those times are lower bounds, so the test*understates* the true difference. |

Always read `summary_by_solver.csv` (success rates) alongside
`statistical_tests.csv`.

## Why energy is not in the cross-solver tests

`energy` and anything derived from it are **refused** for cross-solver tests
(`note = not_comparable_across_solvers`). A QUBO solver reports its own
normalized, per-window-summed Hamiltonian value; ILP reports "timesteps away
from goal"; CBS reports sum-of-costs. Those are unrelated quantities.
Moreover a QUBO Hamiltonian has no meaningful zero, each penalty term drops
its additive constant, so energy is an *interval* scale, not a ratio scale:
differences are meaningful, ratios and percentages are not. For energy, use
the within-solver `energy_excess` column (`compute_energy_excess`); for
cross-solver solution quality use `avg_path_efficiency` / `min_path_efficiency`,
which come from decoded paths and are commensurable.

---

# Energy-landscape diagnostics

`analysis/aggregate.py::compute_energy_diagnostics` → **`energy_diagnostics.csv`**.
One row per *energy class* — `(instance_map, problem_name, solver_name, penalty_set, preprocess)` — describing what that configuration's energies
actually tell you. Built for stochastic backends, where a single run's energy
says little and the shape of the distribution over repeated runs is what
matters.

| Column(s)                                                                             | Reads as                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_energy_runs`                                                                     | Valid runs that produced a solver energy. Can be`0` while the success rate is 100%. BFS pre-processing can solve every window and the solver never runs.                                                                                                                                                                                                                                                                                                                           |
| `best_energy`, `median_energy`, `energy_iqr`, `median_excess`, `max_excess` | The spread, in the solver's own raw units. Tight ⇒ the sampler lands in the same basin repeatedly; long tail ⇒ it doesn't — a`num_reads` / `layers` / `opt_steps` problem, not a formulation one. **Never compare these between configurations.**                                                                                                                                                                                                                       |
| `hit_rate_best`                                                                     | Fraction of valid runs reaching the best energy found. The direct "how often does it actually work" number for a stochastic solver.                                                                                                                                                                                                                                                                                                                                                    |
| `separation_auc`, `n_inversions`, `separated`                                   | The sharp penalty-balance test.`separation_auc = P(invalid energy > valid energy)`, ties = ½. An invalid run's energy is a well-defined evaluation of the same Hamiltonian; if any invalid bitstring scores *below* a valid one (`n_inversions > 0`, `separated = False`), the penalty enforcing the violated constraint is provably too weak (this is exactly the `K_crash ≤ K_adj` relationship). One inversion is one counterexample — no calibration judgment needed. |
| `quality_rho`, `quality_p`, `quality_n`                                         | Spearman rank correlation between energy and path efficiency across the config's valid runs. Lower energy is better, higher efficiency is better, so a well-balanced QUBO gives a**negative** `rho`. Around zero (or positive) means the energy landscape isn't tracking solution quality and the penalty set needs recalibrating. Rank-based on purpose: invariant to the additive constant and `normalize_scale`.                                                          |
| `windows_consistent`                                                                | `False` when the config's runs used differing window counts — energies then carry different dropped constants and aren't sound to compare even within the configuration.                                                                                                                                                                                                                                                                                                            |
| `median_reduction_ratio`, `n_solver_skipped`                                      | How much of the "result" is really BFS.`≥ 0.9` means the rows are mostly measuring pre-processing; run the `preprocess` ablation to separate solver from BFS.                                                                                                                                                                                                                                                                                                                     |
| `note`                                                                              | Plain-text summary of whichever of the above tripped.                                                                                                                                                                                                                                                                                                                                                                                                                                  |

---

# Failure causes

`analysis/aggregate.py::compute_failure_causes` → **`failure_causes.csv`**.
Invalid runs per `(instance, problem, solver, preprocess)`, split by
`invalid_cause`:

- **`pre_processing`** — BFS or diagonal fixing pinned the robots into a
  conflict *before* the solver was invoked. No amount of solver tuning or
  penalty reweighting recovers this; the variables needed to avoid it were
  already removed. Fix the formulation / windowing.
- **`solver_sampling`** — the solver returned a bitstring that violates a
  penalty that *was* present in the Hamiltonian. A sampler or
  penalty-weight problem. Fix the solver params or the penalty set.

This is the first question to ask of any failing cell.

---

# Analysis CLIs

All run from the repo root. Every step is also a plain importable function
(`aggregate_sweep`, `generate_all_plots`, `export_benchmark_table`, …) so
FastAPI can call them directly.

```bash
# JSON → the six CSVs under <sweep_dir>/analysis/
python -m quantum.benchmark.analysis.run_aggregate -d results/sweeps/<sweep_id>

# Plotly figures (HTML + PNG) from runs_long.csv
python -m quantum.benchmark.analysis.run_plots -d results/sweeps/<sweep_id>

# aggregate → plots → LaTeX table, in one go
python -m quantum.benchmark.analysis.run_report -d results/sweeps/<sweep_id>

# QUBO size per pre-processing mode WITHOUT solving — covers instances no
# backend can finish. `raw` = unpruned windowed baseline that every reduction
# ratio is measured against; `encoded_variables` = the actual problem size.
python -m quantum.benchmark.analysis.run_variable_scaling \
    --config sweep_configs/classical_test.yaml -o results/variable_scaling.csv

# push the CSVs + manifest + .tex (not the raw JSON, not the plots) to a
# Hugging Face dataset repo for the FastAPI Space to serve
python -m quantum.benchmark.analysis.publish --repo user/name results/sweeps/<sweep_id>
```
